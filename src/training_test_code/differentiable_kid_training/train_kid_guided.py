"""Full-sampler KID gradients with bounded replay memory and RGB supervision.

Training entry points require an initialized GPU_CONFIG. Settings are passed as
function parameters, and model imports are deferred until training starts.
"""
import gc
import hashlib
import inspect
import json
import math
import signal
import threading
import time
from pathlib import Path
import torch
from .sampling import EulerSampler, shifted_grid, random_grid, noise_bank
from .kid_loss import kid_and_gradient, self_constant
from .gradient_cache import (collect_features, replay_backward, snapshot_gradients,
                             combine_gradients, norm, term_gradient_norms, verify_cached_gradient)
from .features import FloatInception, reference_features
from .clean_features import CleanFloatInception
from .rgb_pairs import RGBPairs
from .reporting import Logs, atomic_save, file_hash, implementation_hash, write_json, save_checkpoint, memory_gb, montage
from .validation import evaluate


class MemoryBudgetExceeded(RuntimeError):
    pass


class VelocityCall:
    """Compile just the velocity model, with an explicit eager fallback."""
    def __init__(self, model, enabled, mode):
        self.model = model
        self.compiled = None
        if enabled:
            try:
                self.compiled = torch.compile(model, mode=mode)
            except Exception as exc:
                print(f'Velocity compiler unavailable; using eager execution: {exc}',flush=True)

    def disable(self):
        self.compiled = None

    def __call__(self, x, t):
        # Checkpoint uses an internal exception to terminate recomputation.
        # Never intercept exceptions inside the checkpointed function. Genuine
        # execution failures are handled after the update has unwound instead.
        return self.model(x,t) if self.compiled is None else self.compiled(x,t)


def _compatible_execution_patch(previous, current, manifest_path=None):
    """Allow pinned legacy-metric source migrations; keep objective/data checks."""
    path = Path(manifest_path) if manifest_path is not None else Path(__file__).with_name('compatibility.json')
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding='utf-8'))
    old = json.loads(json.dumps(previous,sort_keys=True,default=str))
    new = json.loads(json.dumps(current,sort_keys=True,default=str))
    allowed = manifest.get('allowed_package_hash_changes',{})
    old_sources,new_sources = old.get('package_source',{}),new.get('package_source',{})
    additions = manifest.get('allowed_package_additions',{})
    if set(old_sources)-set(new_sources):
        return False
    for name in set(new_sources)-set(old_sources):
        if additions.get(name) != new_sources[name]:
            return False
        old_sources[name] = new_sources[name]
    for name, value in manifest.get('legacy_objective_defaults',{}).items():
        old.setdefault('objective',{}).setdefault(name,value)
    changed = [name for name in old_sources if old_sources[name] != new_sources[name]]
    if not changed:
        return False
    for name in changed:
        hashes = allowed.get(name,{})
        predecessors = hashes.get('from',[])
        predecessors = [predecessors] if isinstance(predecessors,str) else predecessors
        if old_sources[name] not in predecessors or new_sources[name] != hashes.get('to'):
            return False
        old_sources[name] = new_sources[name]
    return old == new


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _clear(model, device):
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def TrainDifferentiableKID(
        model_path: str,
        pair_seeds: list[int] | tuple[int, ...] = (100001,100002,100003),
        pair_cache_folder: str = 'cache/training_pairs',
        reference_folder: str = 'data/cifar_official/fid_reference_flat',
        output_folder: str = 'outputs/differentiable_kid_training/Leg_A_2',
        max_hours: float = 10.0,
        max_updates: int = 10000,
        sample_count: int = 2048,
        microbatch_size: int = 8,
        max_microbatch_size: int = 128,
        memory_budget_gb: float = 40.0,
        target_peak_gb: float = 32.0,
        generation_steps: int = 32,
        generation_min_time: float = .001,
        generation_start_time: float = 1.0,
        rgb_num_times: int = 128,
        rgb_batch_size_per_time: int = 2,
        rgb_grad_accum_steps: int = 3,
        rgb_time_chunk_size: int = 16,
        rgb_min_time: float = .001,
        rgb_max_time: float = .999,
        rgb_weight: float | None = None,
        rgb_gradient_fraction: float = .1,
        learning_rate: float = 1e-6,
        warmup_updates: int = 10,
        adam_betas: tuple[float,float] = (.9,.999),
        adam_epsilon: float = 1e-8,
        gradient_clip_norm: float = 1.0,
        kid_loss_scale: float = 1000.0,
        kernel_block_size: int = 512,
        reference_batch_size: int = 64,
        validation_samples: int = 2048,
        validation_every: int = 10,
        fid_every: int = 50,
        alternative_grid_every: int = 50,
        alternative_grid_steps: int = 40,
        baseline_reference_steps: int = 128,
        comparison_cutoff: float = .09,
        confirmation_samples: int = 8192,
        kid_subsets: int = 100,
        kid_subset_size: int = 1000,
        patience_evaluations: int = 20,
        seed: int = 710001,
        validation_seed: int = 810001,
        confirmation_seed: int = 910001,
        inception_weights_path: str | None = None,
        run_gradient_checks: bool = True,
        resume: bool = True,
        use_clean_fid: bool = True):
    """Train for up to max_hours per invocation and resume from latest.pth.

    Inputs are FP32, as configured centrally. The source model is never saved
    over. All metric-training noises and time grids vary between updates.
    max_hours is checked between operations, not a hard interrupt of CUDA work.
    use_clean_fid selects clean-preprocessing KID training and clean FID/KID
    monitoring. False retains the original TensorFlow-compatible metric path.
    """
    started = time.monotonic()
    from src.core.globals import GPU_CONFIG
    from src.modules.velocity_model import VelocityModel
    if not torch.is_grad_enabled():
        raise RuntimeError('Call training with autograd enabled.')
    if not isinstance(use_clean_fid,bool):
        raise TypeError('use_clean_fid must be a bool.')
    device = torch.device(GPU_CONFIG.device)
    if GPU_CONFIG.model_dtype != torch.float32 or GPU_CONFIG.use_autocast:
        raise ValueError('Initialize GPU_CONFIG with PRECISE: FP32 weights and no autocast.')
    positive = [max_updates,sample_count,microbatch_size,max_microbatch_size,generation_steps,
                rgb_num_times,rgb_batch_size_per_time,rgb_grad_accum_steps,rgb_time_chunk_size,
                validation_samples,validation_every,fid_every,alternative_grid_every,kernel_block_size,
                reference_batch_size,kid_subsets,kid_subset_size]
    if min(positive) < 1 or sample_count < 2 or validation_samples < 10 or max_hours <= 0:
        raise ValueError('Invalid training or validation sizes/budget.')
    if not 0 < target_peak_gb < memory_budget_gb or learning_rate <= 0 or kid_loss_scale <= 0:
        raise ValueError('Invalid memory budget or optimization scale.')
    if rgb_weight is not None and rgb_weight < 0 or rgb_gradient_fraction < 0:
        raise ValueError('RGB weights must be nonnegative.')
    shifted_grid(generation_steps,generation_min_time,.5,generation_start_time)
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    folder = Path(output_folder)
    folder.mkdir(parents=True,exist_ok=True)
    latest = folder/'latest.pth'
    previous_configuration = {}
    if (folder/'configuration.json').is_file():
        previous_configuration = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    if latest.exists():
        if not resume:
            raise FileExistsError(f'{latest} exists. Resume it or choose a new output folder.')
        if previous_configuration.get('use_clean_fid',False) != use_clean_fid:
            raise RuntimeError('This output folder uses a different metric protocol. '
                               'Use a new output folder to switch protocols, or keep '
                               'use_clean_fid=False when resuming an original run.')
    if source.resolve() in {(folder/name).resolve() for name in ['latest.pth','best_kid.pth','best_fid.pth']}:
        raise ValueError('Output checkpoint would overwrite the input model.')
    logs = Logs(folder)
    # This dictionary is persisted metadata, never forwarded as pipeline inputs.
    arguments = locals().copy()
    configuration = {name:arguments[name] for name in inspect.signature(TrainDifferentiableKID).parameters}
    source_files = sorted(Path(__file__).parent.glob('*.py'))
    identity_data = {
        'source_checkpoint':file_hash(source),
        'package_source':{p.name:implementation_hash(p) for p in source_files},
        'pair_files':[(str(Path(pair_cache_folder)/f'cifar_fixed_pairs_{s}.pt'),
                       (Path(pair_cache_folder)/f'cifar_fixed_pairs_{s}.pt').stat().st_size,
                       (Path(pair_cache_folder)/f'cifar_fixed_pairs_{s}.pt').stat().st_mtime_ns) for s in pair_seeds],
        'objective':{k:configuration[k] for k in [
            'sample_count','generation_steps','generation_min_time','generation_start_time',
            'rgb_num_times','rgb_batch_size_per_time','rgb_grad_accum_steps','rgb_min_time','rgb_max_time',
            'rgb_weight','rgb_gradient_fraction','learning_rate','warmup_updates','adam_betas','adam_epsilon',
            'gradient_clip_norm','kid_loss_scale','seed','validation_seed','confirmation_seed',
            'validation_samples','kid_subsets','kid_subset_size','use_clean_fid']},
        'torch':torch.__version__, 'device':str(device),
        'model_source':file_hash(inspect.getfile(VelocityModel)),
        'gpu_profile':str(getattr(GPU_CONFIG,'profile','')), 'model_dtype':str(GPU_CONFIG.model_dtype)}
    print(f'Loading {source}; outputs: {folder}. Time budget {max_hours:g} hours.',flush=True)
    model = VelocityModel.load(str(source),compile=False)
    if model is None:
        raise RuntimeError('VelocityModel.load returned None.')
    model.to(device=device,dtype=torch.float32).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Expected FP32 model parameters.')
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f'Learned coefficients: {sum(p.numel() for p in parameters):,}',flush=True)
    protocol = 'clean-fid/clean (PIL bicubic)' if use_clean_fid else 'torch-fidelity/inception-v3-compat'
    print(f'Metric protocol: {protocol}. IS uses torch-fidelity/inception-v3-compat.',flush=True)
    logs.event('metric_protocol',use_clean_fid=use_clean_fid,kid_fid=protocol,
               is_protocol='torch-fidelity/inception-v3-compat')
    extractor = (CleanFloatInception if use_clean_fid else FloatInception)(device,inception_weights_path)
    extractor.verify(device)
    real, reference_id = reference_features(reference_folder,extractor,device,folder/'feature_cache',reference_batch_size)
    identity_data['reference'] = reference_id
    identity = hashlib.sha256(json.dumps(identity_data,sort_keys=True,default=str).encode()).hexdigest()
    reference_cpu = real.cpu()
    from torch_fidelity.metric_fid import fid_features_to_statistics
    real_statistics = fid_features_to_statistics(reference_cpu)
    constant_path = folder/'feature_cache'/f'constant_{reference_id[:24]}.pt'
    if constant_path.exists():
        constant_data = torch.load(constant_path,map_location='cpu',weights_only=True)
        if constant_data['identity'] != reference_id:
            raise RuntimeError('Reference constant identity mismatch.')
        real_constant = constant_data['value']
    else:
        print('Computing the real KID constant once...',flush=True)
        real_constant = self_constant(real,kernel_block_size)
        atomic_save({'identity':reference_id,'value':real_constant},constant_path)
    pairs = RGBPairs(pair_seeds,pair_cache_folder,device,rgb_num_times,rgb_batch_size_per_time,
                     rgb_grad_accum_steps,rgb_min_time,rgb_max_time,seed+3000000)
    if any(tuple(pairs.data[s][0].shape[1:]) != (3,32,32) for s in pair_seeds):
        raise ValueError('This experiment expects CIFAR-10 RGB 32x32 caches.')
    effective_target = target_peak_gb
    if device.type == 'cuda':
        free,total = torch.cuda.mem_get_info(device)
        available = (free+torch.cuda.memory_reserved(device))/1e9
        effective_target = min(target_peak_gb,.8*min(memory_budget_gb,available))
        print(f'GPU free {free/1e9:.2f} GB, total {total/1e9:.2f} GB; '
              f'effective allocated-memory target {effective_target:.2f} GB.',flush=True)
        logs.event('memory_budget',available_process_gb=available,configured_budget_gb=memory_budget_gb,
                   configured_target_gb=target_peak_gb,effective_target_gb=effective_target)
        if torch.cuda.memory_allocated(device)/1e9 > effective_target:
            raise MemoryBudgetExceeded('Persistent data/model memory already exceeds the effective target budget.')
    optimizer = torch.optim.AdamW(parameters,lr=learning_rate,betas=adam_betas,eps=adam_epsilon,weight_decay=0)
    update, best_kid, best_fid, stale = 0,float('inf'),float('inf'),0
    microbatch = min(microbatch_size,max_microbatch_size)
    microbatch_cap, rgb_chunk = max_microbatch_size,rgb_time_chunk_size
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity'] != identity:
            previous_identity_path = folder/'identity.json'
            previous_identity = json.loads(previous_identity_path.read_text(encoding='utf-8')) if previous_identity_path.is_file() else {}
            previous_hash = hashlib.sha256(json.dumps(previous_identity,sort_keys=True,default=str).encode()).hexdigest()
            if previous_hash != state['identity'] or not _compatible_execution_patch(previous_identity,identity_data):
                raise RuntimeError('Resume identity differs. Use the original settings/files or a new output folder.')
            logs.event('compatible_execution_patch_resume',previous_identity=state['identity'],identity=identity)
            print('Resuming the previous package checkpoint through the verified compatibility patch.',flush=True)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        pairs.load_state_dict(state['rgb_pairs'])
        update, rgb_weight = state['update'],state['rgb_weight']
        best_kid,best_fid,stale = state['best_kid'],state['best_fid'],state['stale']
        saved_cap = state.get('microbatch_cap',state['microbatch'])
        previous_limit = previous_configuration.get('max_microbatch_size',saved_cap)
        # A configured ceiling is adjustable. A cap reduced by an earlier OOM
        # remains a measured bound for this run.
        microbatch_cap = min(max_microbatch_size,saved_cap) if saved_cap < previous_limit else max_microbatch_size
        microbatch = min(state['microbatch'],microbatch_cap)
        rgb_chunk = state.get('rgb_time_chunk',rgb_chunk)
        del state
        print(f'Resuming update {update}, RGB epoch {pairs.epoch+1}, seed {pairs.current_seed}.',flush=True)
    write_json(folder/'configuration.json',configuration)
    if run_gradient_checks:
        verify_cached_gradient(model,extractor,real,device)
        _clear(model,device)
    mode = getattr(GPU_CONFIG.compile_mode_velocity,'value',str(GPU_CONFIG.compile_mode_velocity))
    enabled = bool(GPU_CONFIG.compile_model and mode != 'none')
    velocity = VelocityCall(model,enabled,mode)
    if run_gradient_checks and enabled:
        print('Checking compiled sampler gradients against the eager reference...',flush=True)
        try:
            verify_cached_gradient(model,extractor,real,device,velocity)
        except torch.OutOfMemoryError as exc:
            exc.__traceback__ = None
            velocity.disable()
            _clear(model,device)
            print('Compiled gradient check exceeded memory; using the verified eager model.',flush=True)
        except Exception as exc:
            velocity.disable()
            _clear(model,device)
            print(f'Compiled gradients failed verification; using the verified eager model: {exc}',flush=True)
    monitor_grid = shifted_grid(generation_steps,generation_min_time,.5,generation_start_time)
    stop_requested = [False]
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig,lambda signum,frame: stop_requested.__setitem__(0,True))
    checkpoint_safe = True
    last_evaluation = -1
    last_eval_seconds, average_update_seconds = 0.0,0.0

    def save(name):
        save_checkpoint(folder/name,model,optimizer,update,pairs,rgb_weight,best_kid,best_fid,
                        identity,microbatch,stale,microbatch_cap,rgb_chunk)
        if name == 'latest.pth':
            # Commit the new identity after latest is atomic. An interruption
            # before that point leaves the old migration metadata available.
            write_json(folder/'identity.json',identity_data)

    def validation(label,grid,count,bank_seed,with_fid):
        nonlocal microbatch,microbatch_cap,last_eval_seconds
        begin = time.monotonic()
        while True:
            try:
                metrics,pictures = evaluate(velocity,extractor,reference_cpu,real_statistics,grid,device,
                    count,microbatch,bank_seed,kid_subsets,kid_subset_size,with_fid,use_clean_fid)
                break
            except torch.OutOfMemoryError as exc:
                exc.__traceback__ = None
                if microbatch == 1:
                    raise
                microbatch = max(1,microbatch//2)
                microbatch_cap = min(microbatch_cap,microbatch)
                _clear(model,device)
                print(f'Validation OOM: retrying unchanged bank with batch {microbatch}.',flush=True)
        if any(v is not None and not math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite official validation metric.')
        last_eval_seconds = time.monotonic()-begin
        logs.append('validation.csv',{'update':update,'label':label,'samples':count,
            'noise_seed':bank_seed,'steps':len(grid)-1,'terminal_time':float(grid[-2]),
            'kid':metrics['kid'],'kid_subset_std':metrics['kid_subset_std'],
            'fid':metrics['fid'],'is':metrics['is'],'seconds':last_eval_seconds})
        montage(pictures,folder/f'samples_{label}_{update:06d}.png')
        print(f'Validation {label}, update {update} [{protocol}]: KID {metrics["kid"]:.7f}, '
              f'subset SD {metrics["kid_subset_std"]:.7f}, FID {metrics["fid"]}, IS {metrics["is"]}',flush=True)
        return metrics

    reason = 'update_limit'
    try:
        initial = validation('monitor',monitor_grid,validation_samples,validation_seed,True)
        if initial['kid'] < best_kid:
            best_kid = initial['kid']
            save('best_kid.pth')
        if initial['fid'] < best_fid:
            best_fid = initial['fid']
            save('best_fid.pth')
        last_evaluation = update
        save('latest.pth')
        # The old larger cutoff is a comparison, never the new training target.
        if update == 0 and comparison_cutoff > generation_min_time:
            comparison = shifted_grid(30,comparison_cutoff,.5,generation_start_time)
            validation('cutoff_comparison',comparison,validation_samples,validation_seed,False)
        if update == 0 and baseline_reference_steps >= 3:
            reference_grid = shifted_grid(baseline_reference_steps,generation_min_time,.5,generation_start_time)
            validation('fine_grid_reference',reference_grid,validation_samples,validation_seed,False)
        while update < max_updates:
            elapsed = time.monotonic()-started
            reserve = max(300.0,2*last_eval_seconds,1.5*average_update_seconds)
            if stop_requested[0] or elapsed+reserve >= max_hours*3600:
                reason = 'stop_requested' if stop_requested[0] else 'time_budget'
                break
            if patience_evaluations > 0 and stale >= patience_evaluations:
                reason = 'validation_patience'
                break
            step_start = time.monotonic()
            grid,phase = random_grid(generation_steps,generation_min_time,seed+update*10+1,generation_start_time)
            noises = noise_bank(sample_count,3,32,seed+update*10+2)
            rgb_batches = pairs.prepare()
            step_epoch,step_seed = pairs.epoch+1,pairs.current_seed
            while True:
                cached = feature_gradient = metric_grads = None
                model.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                sampler = EulerSampler(velocity,grid,device,checkpoint_steps=True)
                try:
                    begin = time.monotonic()
                    cached,clipped = collect_features(sampler,extractor,noises,device,microbatch)
                    _synchronize(device)
                    collect_seconds = time.monotonic()-begin
                    begin = time.monotonic()
                    kid,feature_gradient = kid_and_gradient(cached,real,real_constant,kernel_block_size,kid_loss_scale)
                    _synchronize(device)
                    kernel_seconds = time.monotonic()-begin
                    begin = time.monotonic()
                    replay_error = replay_backward(sampler,extractor,noises,cached,feature_gradient,device,microbatch)
                    _synchronize(device)
                    replay_seconds = time.monotonic()-begin
                    metric_grads = snapshot_gradients(parameters)
                    del cached,feature_gradient
                    cached = feature_gradient = None
                    model.zero_grad(set_to_none=True)
                    begin = time.monotonic()
                    rgb_loss = pairs.backward(model,rgb_batches,rgb_chunk)
                    _synchronize(device)
                    rgb_seconds = time.monotonic()-begin
                    if rgb_weight is None:
                        km,kr = norm(metric_grads),norm([p.grad for p in parameters])
                        if not math.isfinite(km) or not math.isfinite(kr) or km == 0:
                            raise FloatingPointError('Cannot calibrate RGB weight from invalid or zero metric gradients.')
                        rgb_weight = rgb_gradient_fraction*km/max(kr,1e-30)
                        print(f'Calibrated fixed RGB weight: {rgb_weight:.6g}',flush=True)
                    metric_norm,rgb_norm,cosine = combine_gradients(parameters,metric_grads,rgb_weight)
                    metric_grads = None
                    peak = memory_gb(device)
                    if peak > effective_target:
                        raise MemoryBudgetExceeded(f'Peak {peak:.2f} GB exceeds target {effective_target:.2f} GB.')
                    break
                except (torch.OutOfMemoryError,MemoryBudgetExceeded) as exc:
                    message = str(exc)
                    exc.__traceback__ = None
                    cached = feature_gradient = metric_grads = None
                    _clear(model,device)
                    if microbatch == 1 and rgb_chunk == 1:
                        raise RuntimeError('Cannot fit the unchanged objective even at minimum chunk sizes.') from exc
                    microbatch = max(1,microbatch//2)
                    rgb_chunk = max(1,rgb_chunk//2)
                    microbatch_cap = min(microbatch_cap,microbatch)
                    logs.event('memory_backoff',update=update,microbatch=microbatch,rgb_chunk=rgb_chunk,error=message)
                    print(f'Memory backoff: batch {microbatch}, RGB time chunk {rgb_chunk}; retrying same update.',flush=True)
                except Exception as exc:
                    compiler_error = isinstance(exc,RuntimeError) or type(exc).__module__.startswith(('torch._dynamo','torch._inductor'))
                    if not compiler_error or velocity.compiled is None:
                        raise
                    message = f'{type(exc).__name__}: {exc}'
                    exc.__traceback__ = None
                    cached = feature_gradient = metric_grads = None
                    velocity.disable()
                    _clear(model,device)
                    logs.event('compile_update_fallback',update=update,error=message)
                    print(f'Compiled update failed; retrying the unchanged update eagerly: {message}',flush=True)
            term_norms = term_gradient_norms(parameters,getattr(model,'degree',None))
            unclipped = float(torch.nn.utils.clip_grad_norm_(parameters,gradient_clip_norm if gradient_clip_norm > 0 else float('inf'),error_if_nonfinite=True))
            current_lr = learning_rate*min(1.0,(update+1)/max(1,warmup_updates))
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            before_step = [p.detach().clone() for p in parameters]
            weight_norm = norm(before_step)
            # An optimizer failure may be non-atomic. Never save a potentially partial step as latest.
            checkpoint_safe = False
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError('Optimizer produced nonfinite parameters; previous checkpoints are intact.')
            step_norm = norm([p.detach()-old for p,old in zip(parameters,before_step)])
            del before_step
            pairs.commit()
            update += 1
            checkpoint_safe = True
            model.zero_grad(set_to_none=True)
            _synchronize(device)
            duration = time.monotonic()-step_start
            average_update_seconds = duration if not average_update_seconds else .8*average_update_seconds+.2*duration
            logs.append('training.csv',{'update':update,'rgb_epoch':step_epoch,'rgb_seed':step_seed,
                'phase':phase,'kid_float':kid,'rgb_mse':rgb_loss,'rgb_weight':rgb_weight,
                'metric_grad_norm':metric_norm,'rgb_grad_norm':rgb_norm,'gradient_cosine':cosine,
                'parameter_step_norm':step_norm,'relative_parameter_step':step_norm/max(weight_norm,1e-30),
                'unclipped_combined_norm':unclipped,'lr':current_lr,'clipped_pixel_fraction':clipped,
                'replay_error':replay_error,'microbatch':microbatch,'rgb_time_chunk':rgb_chunk,
                'peak_allocated_gb':memory_gb(device),
                'peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1e9 if device.type == 'cuda' else 0.0,
                'collection_seconds':collect_seconds,
                'kernel_seconds':kernel_seconds,'replay_seconds':replay_seconds,
                'rgb_seconds':rgb_seconds,'update_seconds':duration,'elapsed_hours':(time.monotonic()-started)/3600})
            logs.event('grid',update=update,times=grid.tolist())
            logs.event('rgb_time_bins',update=update,mse=pairs.last_time_bins)
            logs.event('coefficient_gradient_norms',update=update,norms=term_norms)
            print(f'Update {update}/{max_updates} | RGB epoch {step_epoch} seed {step_seed} | '
                  f'float KID {kid:.7f} | RGB {rgb_loss:.5f} | cosine {cosine:+.3f} | '
                  f'{duration:.1f}s | peak {memory_gb(device):.2f} GB | batch {microbatch}',flush=True)
            if update % validation_every == 0:
                metrics = validation('monitor',monitor_grid,validation_samples,validation_seed,update % fid_every == 0)
                last_evaluation = update
                if metrics['kid'] < best_kid:
                    best_kid,stale = metrics['kid'],0
                    save('best_kid.pth')
                else:
                    stale += 1
                if metrics['fid'] is not None and metrics['fid'] < best_fid:
                    best_fid = metrics['fid']
                    save('best_fid.pth')
                save('latest.pth')
            if update % alternative_grid_every == 0:
                alternative = shifted_grid(alternative_grid_steps,generation_min_time,.173,generation_start_time)
                validation('unseen_grid',alternative,validation_samples,validation_seed,False)
            if microbatch < microbatch_cap and memory_gb(device) < .55*effective_target:
                microbatch = min(2*microbatch,microbatch_cap)
                print(f'Next update replay batch: {microbatch} (memory target {effective_target:.2f} GB).',flush=True)
        if last_evaluation != update and not stop_requested[0]:
            metrics = validation('monitor',monitor_grid,validation_samples,validation_seed,True)
            if metrics['kid'] < best_kid:
                best_kid = metrics['kid']
                save('best_kid.pth')
            if metrics['fid'] < best_fid:
                best_fid = metrics['fid']
                save('best_fid.pth')
        save('latest.pth')
        remaining = max_hours*3600-(time.monotonic()-started)
        estimated_confirmation = max(last_eval_seconds,1)*confirmation_samples/validation_samples*1.5
        if confirmation_samples >= 10 and not stop_requested[0] and remaining > estimated_confirmation+60:
            # Confirm the selected model, then restore latest weights before any
            # error handler can save them alongside the latest optimizer state.
            latest_weights = torch.load(latest,map_location='cpu',weights_only=False)['state_dict']
            best_weights = torch.load(folder/'best_kid.pth',map_location='cpu',weights_only=False)['state_dict']
            checkpoint_safe = False
            try:
                model.load_state_dict(best_weights)
                validation('confirmation_best_kid',monitor_grid,confirmation_samples,confirmation_seed,True)
            finally:
                model.load_state_dict(latest_weights)
                checkpoint_safe = True
            del latest_weights,best_weights
        result = {'reason':reason,'completed_updates':update,'best_monitor_kid':best_kid,
                  'best_monitor_fid':best_fid,'elapsed_hours':(time.monotonic()-started)/3600,
                  'latest_checkpoint':str(latest),'output_folder':str(folder)}
        write_json(folder/'summary.json',result)
        print(f'Training finished: {result}',flush=True)
        return result
    except BaseException as exc:
        logs.event('stopped_with_error',update=update,error_type=type(exc).__name__,error=str(exc),checkpoint_safe=checkpoint_safe)
        if checkpoint_safe:
            save('latest.pth')
        raise
    finally:
        for sig,handler in previous_handlers.items():
            signal.signal(sig,handler)


def TrainTheModel():
    """Call this after GPU_CONFIG.Initialize(profile=HardwareProfile.PRECISE)."""
    model_name = "CD"
    use_clean: bool = True
    

    model_path = f"saved_models/{model_name}.pth"
    output_folder = f"outputs/differentiable_kid_training/{model_name}"
    if(use_clean):
        output_folder += "_clean"
    
    return TrainDifferentiableKID(
        model_path=model_path,
        pair_seeds=[100001,100002,100003],
        seed = 410001,
        pair_cache_folder='cache/training_pairs',
        reference_folder='data/cifar_official/fid_reference_flat',
        output_folder=output_folder,
        max_hours=30.0,
        max_updates=10000,
        sample_count=2048,
        microbatch_size=8,
        max_microbatch_size=128,
        memory_budget_gb=40.0,
        target_peak_gb=32.0,
        generation_steps=32,
        generation_min_time=.001,
        rgb_num_times=128,
        rgb_batch_size_per_time=2,
        rgb_grad_accum_steps=3,
        rgb_time_chunk_size=16,
        rgb_weight=None,
        rgb_gradient_fraction=.1,
        learning_rate=1e-6,
        validation_samples=2048,
        validation_every=10,
        fid_every=50,
        alternative_grid_every=50,
        confirmation_samples=8192,
        patience_evaluations=200,
        run_gradient_checks=True,
        resume=True,
        use_clean_fid=use_clean)

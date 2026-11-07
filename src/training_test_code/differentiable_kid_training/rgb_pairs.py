"""The b2 datasets, paired flips, and complete-epoch seed rotation."""
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


class RGBPairs:
    def __init__(self, seeds, cache_folder, device, num_times=128, images_per_time=2,
                 accum_steps=3, minimum=0.001, maximum=0.999, seed=81001):
        if not seeds or min(num_times, images_per_time, accum_steps) < 1:
            raise ValueError('Provide seeds and positive RGB batching settings.')
        if not 0 < minimum < maximum <= 1:
            raise ValueError('Invalid RGB time interval.')
        self.seeds, self.device = list(seeds), torch.device(device)
        self.num_times, self.images_per_time, self.accum_steps = num_times, images_per_time, accum_steps
        self.minimum, self.maximum, self.seed = minimum, maximum, seed
        self.batch_size = num_times*images_per_time
        self.data = {}
        for cache_seed in self.seeds:
            path = Path(cache_folder)/f'cifar_fixed_pairs_{cache_seed}.pt'
            print(f'Loading paired RGB cache: {path}', flush=True)
            if self.device.type == 'cuda':
                free,_ = torch.cuda.mem_get_info(self.device)
                if free < 4*path.stat().st_size:
                    raise RuntimeError(f'Insufficient free VRAM for safe loading/mirroring of {path}.')
            data = torch.load(path, map_location=self.device, weights_only=True)
            g, n = data['G'], data['N']
            if g.shape != n.shape or g.ndim != 4 or g.dtype != torch.float32 or n.dtype != torch.float32:
                raise ValueError(f'{path}: expected matching FP32 NCHW G,N tensors.')
            if not torch.isfinite(g).all() or not torch.isfinite(n).all():
                raise ValueError(f'{path}: nonfinite cached pairs.')
            if self.device.type == 'cuda':
                free,_ = torch.cuda.mem_get_info(self.device)
                required = 3*(g.numel()*g.element_size()+n.numel()*n.element_size())
                if free < required:
                    raise RuntimeError(f'Insufficient free VRAM for mirroring {path}; pairs were not altered.')
            self.data[cache_seed] = (torch.cat((g, torch.flip(g, dims=[3]))),
                                     torch.cat((n, torch.flip(n, dims=[3]))))
            print(f'  {cache_seed}: {len(g)} original, {2*len(g)} mirrored pairs; '
                  f'G range [{float(g.min()):.3f}, {float(g.max()):.3f}], N std {float(n.std()):.3f}', flush=True)
            del data, g, n
        self.epoch, self.update_in_epoch, self.permutation = 0, 0, None

    @property
    def current_seed(self):
        return self.seeds[self.epoch % len(self.seeds)]

    @property
    def updates_per_epoch(self):
        return len(self.data[self.current_seed][0])//self.batch_size//self.accum_steps

    def prepare(self):
        g, n = self.data[self.current_seed]
        if self.updates_per_epoch < 1:
            raise ValueError('RGB cache is smaller than one accumulated update.')
        if self.permutation is None:
            rng = torch.Generator(device=self.device).manual_seed(self.seed+self.epoch)
            self.permutation = torch.randperm(len(g), device=self.device, generator=rng)
        batches = []
        for a in range(self.accum_steps):
            batch_index = self.update_in_epoch*self.accum_steps+a
            start = batch_index*self.batch_size
            indices = self.permutation[start:start+self.batch_size]
            rng = torch.Generator(device=self.device).manual_seed(self.seed+1000003*self.epoch+batch_index+10000)
            u = (torch.arange(self.num_times, device=self.device, dtype=torch.float32)
                 + torch.rand(self.num_times, device=self.device, generator=rng))/self.num_times
            times = (self.minimum+(self.maximum-self.minimum)*u).flip(0).reshape(-1, 1)
            batches.append((indices, times))
        return batches

    def backward(self, model, batches, time_chunk_size=16):
        g, n = self.data[self.current_seed]
        predict = torch.vmap(model, in_dims=(0, 0), randomness='error')
        total = 0.0
        bin_sum = torch.zeros(8,device=self.device,dtype=torch.float64)
        bin_count = torch.zeros(8,device=self.device,dtype=torch.float64)
        for indices, times in batches:
            shape = (self.num_times, self.images_per_time, *g.shape[1:])
            gg, nn = g[indices].reshape(shape), n[indices].reshape(shape)
            x = gg + times.reshape(-1, 1, 1, 1, 1)*(nn-gg)
            for start in range(0, self.num_times, time_chunk_size):
                stop = min(start+time_chunk_size, self.num_times)
                fraction = (stop-start)/self.num_times/self.accum_steps
                with sdpa_kernel(SDPBackend.MATH):
                    predicted = predict(x[start:stop], times[start:stop])
                    per_time = (predicted-(nn-gg)[start:stop]).square().mean(dim=(1,2,3,4))
                    loss = per_time.mean()
                    (loss*fraction).backward()
                total += float(loss.detach())*fraction
                bins = (times[start:stop].flatten()*8).long().clamp(0,7)
                bin_sum.scatter_add_(0,bins,per_time.detach().double())
                bin_count.scatter_add_(0,bins,torch.ones_like(per_time,dtype=torch.float64))
        self.last_time_bins = (bin_sum/bin_count.clamp_min(1)).cpu().tolist()
        return total

    def commit(self):
        self.update_in_epoch += 1
        if self.update_in_epoch >= self.updates_per_epoch:
            self.epoch += 1
            self.update_in_epoch, self.permutation = 0, None

    def state_dict(self):
        return {'epoch': self.epoch, 'update_in_epoch': self.update_in_epoch,
                'permutation': None if self.permutation is None else self.permutation.cpu()}

    def load_state_dict(self, state):
        self.epoch, self.update_in_epoch = int(state['epoch']), int(state['update_in_epoch'])
        permutation = state['permutation']
        self.permutation = None if permutation is None else permutation.to(self.device)
        count = len(self.data[self.current_seed][0])
        if self.permutation is not None and len(self.permutation) != count:
            raise ValueError('RGB cache size differs from saved permutation.')
        if not 0 <= self.update_in_epoch < self.updates_per_epoch:
            raise ValueError('Invalid saved RGB epoch position.')

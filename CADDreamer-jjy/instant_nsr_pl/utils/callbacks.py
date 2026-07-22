import os
import subprocess
import shutil
from utils.misc import dump_config, parse_version


import pytorch_lightning
if parse_version(pytorch_lightning.__version__) > parse_version('1.8'):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1
    
    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}")


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
    
    def get_file_list(self):
        # Only snapshot source files. Never recurse into generated / heavy dirs,
        # otherwise each run copies all previous snapshots -> exponential blowup.
        _EXCLUDE_PREFIXES = (
            'exp/', 'neus/exp/', 'ckpts/', 'test_outputs/', 'test_output/',
            'cached_output/', 'our_inputs/', 'finetuning_vae_normal/',
            'sam_pt/', 'pyransac/', '.git/', '__pycache__/',
            'neus/temp_mid_outputs/', 'build_dataset/step2_dsine_dataset/assets/',
        )
        _EXCLUDE_SUFFIXES = ('.pth', '.ckpt', '.png', '.jpg', '.npy', '.obj', '.ply', '.step', '.pyc')
        try:
            files = [
                b.decode() for b in
                set(subprocess.check_output('git ls-files', shell=True).splitlines()) |
                set(subprocess.check_output('git ls-files --others --exclude-standard', shell=True).splitlines())
            ]
        except Exception:
            files = []
        out = []
        for f in files:
            if any(f.startswith(p) or ('/' + p) in f for p in _EXCLUDE_PREFIXES):
                continue
            if f.endswith(_EXCLUDE_SUFFIXES):
                continue
            out.append(f)
        return out
    
    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn("Code snapshot is not saved. Please make sure you have git installed and are in a git repository.")


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, config, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
        self.config = config

    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, 'parsed.yaml'), self.config)
        shutil.copyfile(self.config.cmd_args['config'], os.path.join(self.savedir, 'raw.yaml'))

    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items

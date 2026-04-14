This directory provides compatibility shims for `torch_scatter` and `torch_cluster`
on `linux-aarch64`, where the original compiled extensions were not available via
the tested conda channels. Use it by prepending this directory to `PYTHONPATH`.

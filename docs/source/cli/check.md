# `deeprm check`

Environment diagnostics for Torch / metrics / GPU (prints only problems by default).

## Usage

```{argparse}
   :module: deeprm.utils.check_installation
   :func: parser
   :prog: deeprm check
```
## Examples

```bash
# Show only issues
deeprm check

# Show everything (env dump)
deeprm check --verbose
```

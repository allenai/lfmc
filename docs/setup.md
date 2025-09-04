# Setup

Build the image:

```shell
docker build --platform linux/amd64 -t lfmc .
```

Upload to Beaker:

```shell
if beaker image get "$USER/lfmc" > /dev/null 2>&1; then
    beaker image delete "$USER/lfmc"
fi
beaker image create --workspace "$BEAKER_WORKSPACE" --name lfmc lfmc
```

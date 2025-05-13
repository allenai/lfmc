# Setup

Build the image:

```shell
docker build --platform linux/amd64 -t lfmc .
```

Upload to Beaker:

```shell
if ! beaker image get $USER/lfmc 2> /dev/null; then
    beaker image delete $USER/lfmc;
fi
beaker image create --workspace $WORKSPACE --name lfmc lfmc
```

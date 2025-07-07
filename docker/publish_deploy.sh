docker build --no-cache -f Dockerfile.deploy -t supervisely/rt-detrv2:dev-deploy . && \
docker push supervisely/rt-detrv2:dev-deploy
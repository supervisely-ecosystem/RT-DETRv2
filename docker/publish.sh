# Resolve the latest commit of the SDK test branch so every build re-pulls it (and busts the
# Docker cache when it changed). Override branch with: SDK_BRANCH=<branch> sh docker/publish.sh
SDK_BRANCH="${SDK_BRANCH:-feat/train-resume-upload-tests}"
SDK_REF=$(git ls-remote https://github.com/supervisely/supervisely.git "${SDK_BRANCH}" | cut -f1)
echo "Building with SDK ${SDK_BRANCH} @ ${SDK_REF}"
docker build --build-arg SDK_REF="${SDK_REF}" -t supervisely/rt-detrv2:dev . && \
docker push supervisely/rt-detrv2:dev

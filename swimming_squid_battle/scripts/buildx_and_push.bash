export tag="1.6.2"
export game="swimming_squid_battle"

docker buildx build --platform linux/amd64,linux/arm64 \
-t paiatech/${game}:${tag} -t paiatech/${game}:${tag}-PGE20250312 \
-f ./Dockerfile . --push
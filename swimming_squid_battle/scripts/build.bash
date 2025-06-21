export tag="latest"
export game="swimming_squid_battle"

docker build \
-t ${game}:${tag} \
-f ./Dockerfile .

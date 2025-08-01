# retrieve AWS instance metadata token to test if the build is running on EC2
IMDSV2_TOKEN=`curl -s --connect-timeout 1 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
if [ -z "$IMDSV2_TOKEN" ]; then
    echo "Not running on AWS EC2, skipping AWS-related build steps."
    AWS=false
else
    echo "Running on AWS EC2, proceeding with AWS-related build steps."
    AWS=true
fi

# get script directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")

DOCKER_BUILDKIT=1 docker build --build-arg AWS=$AWS -t dcp:latest ${SCRIPT_DIR}/../../docker
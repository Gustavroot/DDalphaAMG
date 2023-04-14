#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# You must be in the root directory of the DD Alpha AMG project to execute this script.

LOGDIR="testlogs"

function pass() {
    >&2 echo -e "${GREEN}PASS:${NC} $1"
}

# To be called if a configuration file exit abnormally.
function fail() {
    # Print error to stderr
    >&2 echo -e "${RED}!!! TEST FAILURE !!!${NC}"
    >&2 echo -e "Configuration file: $1 --- Error code: $2 "
    >&2 echo -e "Tail of log file $3\n----- begin log -----\n\n"
    >&2 tail $3
    >&2 echo -e "\n\n----- end log -----"
    exit $2
}




# Iterate over different configuration files and try if they exit normally.
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

if [ -z $1 ]; then
    echo "usage: $0 cpu|gpu"
    exit 2
elif [ $1 = "cpu" ]; then
    confs=test/test_confs/cpu_*.ini
elif [ $1 = "gpu" ]; then
    confs=test/test_confs/gpu_*.ini
else
    echo "parameter unsupported!"
    exit 2
fi

for conf in $confs; do
    confname=$(basename -- "$conf")
    logfile="$LOGDIR/${confname%.*}.log"
    ./run -i $conf 2>$logfile >$logfile && pass $conf || fail $conf $? $logfile
done

#!/bin/bash
# GPU Memory Monitor - Run this in terminal 1
# Then run benchmark in terminal 2
# Press Ctrl+C to stop monitoring

OUTPUT_FILE="${1:-/tmp/gpu_memory_log.csv}"
INTERVAL=1

echo "GPU Memory Monitor"
echo "=================="
echo "Output file: $OUTPUT_FILE"
echo "Sampling every ${INTERVAL} second(s)"
echo "Press Ctrl+C to stop"
echo ""

# Create CSV header
echo "timestamp,gpu_name,memory_used_mb,memory_total_mb,utilization_percent" > "$OUTPUT_FILE"

# Monitor loop
while true; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$OUTPUT_FILE"
    sleep $INTERVAL
done

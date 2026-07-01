#!/usr/bin/env bash
# run_imu_siga_sweep.sh — Hipótesis A del handoff 2026-06-30-hito3-ekf-decouple-
# heading-scale: DESACOPLAR heading de escala down-pesando la parte de traslación
# del factor IMU tight (r_v/r_p, del acelerómetro) sin tocar la rotación (r_R, del
# gyro). En imu_information():  σ_v²=σ_p²∝sig_a²  →  subir --imu-sig-a baja el peso
# de la traslación → la IMU aporta SOLO la restricción de rotación (heading) y deja
# la escala al depth-prior in-solver, que ya la ancla en los tramos rectos.
#
# Réplica EXACTA del run ON (strength 10, v_reg 10) del girodemarcado, variando
# SOLO --imu-sig-a: 2 / 10 / 1000 (default es 0.2 = el run ON ancla). Como σ va al
# cuadrado, 0.2→2 = 100× menos peso de traslación, 0.2→10 = 2500×, 0.2→1000 ≈
# rotación-pura. v1/v2/v3 (N=3, el s5 mostró que v1 solo engaña por lotería de seed).
# Anclas OFF y ON (sig_a 0.2) ya existen (2026-06-30). Runtime puro, cero código.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
SCRIPT=scripts/run_sdpvo_metric.py
CAL=third_party/S_DPVO/calib/zed2i_gym_2026-06-30_hd720.txt
CFG=third_party/S_DPVO/config/sweep_p24_ow3_lt6.yaml

run() {  # $1=v  $2=sig_a  $3=tag
  local svo="data/recordings/2026-06-30__rec__u_giro180_gym_girodemarcado_v$1.svo2"
  local imu="results/imu/gym_girodemarcado_v$1"
  local name="gym_girodemarcado_v$1_imu_siga$3"
  echo "########## $name (imu-sig-a $2, strength 10) ##########"
  $PY $SCRIPT --svo "$svo" --calib "$CAL" \
    --config-sdpvo "$CFG" --inject prior_insolver --prior-strength 1000 \
    --depth-mode NEURAL --stride 1 --skip 15 --scale 0.5 --smooth-window 9 \
    --imu "$imu" --imu-strength 10 --imu-sig-a "$2" --imu-v-reg 10 --imu-max-step 2 \
    --name "$name" 2>&1 \
    | grep -E "output:|FPS efectivo|smooth:|ERROR|Traceback|END OF SVO|out of memory|CUDA|clamp|reject"
  echo
}

# valores de --imu-sig-a: por CLI (refinamiento) o el default reproducible {2,10,1000}
SIGAS=("$@"); [ ${#SIGAS[@]} -eq 0 ] && SIGAS=(2 10 1000)
echo "########## SIGA SWEEP  sig_a={${SIGAS[*]}}  N=3 (v1/v2/v3) ##########"
for v in 1 2 3; do
  for sa in "${SIGAS[@]}"; do
    run "$v" "$sa" "$sa"
  done
done
echo "########## SIGA SWEEP DONE ##########"

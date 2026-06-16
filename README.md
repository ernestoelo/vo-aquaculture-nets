# Benchmarking de Odometría Visual para Mallas de Acuicultura — Hito 2 (IPD441)

Entrega de código del Hito 2: implementación y evaluación comparativa de
algoritmos de odometría visual (VO) sobre mallas de acuicultura con una
cámara estéreo ZED 2i, en aire y bajo el agua.

**Autores:** Ernesto Gamero, Fernanda Quintana — UTFSM.
**Informe:** el PDF escrito (formato IEEE-conf) se entrega por separado; este repositorio contiene el código de reproducción y evaluación.

## Estructura

```
dpvo_metric/   NUESTRA modificación de DPVO (escala métrica): patch + src/ + configs
scripts/       scripts de corrida de los 4 modelos (DPVO mono/métrico, MAC-VO prep, ZED PT)
eval/          marco de evaluación (ATE along-track + cierre de lazo)
configs/       ground truth de cintas + configs de corrida por secuencia
NOTICE.md      atribución de terceros (DPVO MIT, MAC-VO, ZED SDK)
```

---

## 1. Qué se evalúa

Cuatro configuraciones sobre la misma cámara y los mismos datos:

| Modelo | Qué es | Escala | fps (x86 RTX 3060) |
|---|---|---|---|
| **DPVO mono** | DPVO monocular [Teed 2023], punto de partida | arbitraria | ~27 |
| **DPVO métrico** | **nuestra modificación**: prior de profundidad estéreo *in-solver* | métrica | ~19 (6.1 embarcado) |
| **MAC-VO** | estado del arte estéreo + covarianza [Qiu 2025, ICRA Best Paper] | métrica | ~1.2 |
| **ZED SDK PT** | *positional tracking* del SDK (visión + IMU), referencia industrial | métrica | tiempo real (Jetson) |

## 2. La modificación: DPVO métrico

DPVO inicializa la profundidad inversa de cada parche **al azar** → escala
ambigua. Nuestra modificación inyecta la profundidad estéreo del SDK de ZED
como **factor unario dentro del bundle adjustment**:

```
E_prior = Σ_p  w_p · ( d_p − 1/Z_zed,p )²
```

sumado al residuo de reproyección. Anclar la profundidad *dentro* de la
optimización (no como reescalado posterior) recupera una escala estable.
**El código de la modificación está en [`dpvo_metric/`](dpvo_metric/)**:
el patch (`dpvo_metric.patch`, 484 líneas sobre `MAC-VO/S_DPVO@f7266f7`),
los archivos modificados completos (`src/`) y los configs. Ver
[`dpvo_metric/README.md`](dpvo_metric/README.md) para dónde está cada cambio.

## 3. Métricas

Calculadas contra el *ground truth* de cintas (cada 1 m) con
`eval/eval_ate_tape_gt.py`. Como el GT es colineal, el alineamiento de
Umeyama de `evo` degenera → se usa una variante **along-track** (proyección
sobre el eje principal). Métricas: **ATE**, **RPE traslacional (1 m)**,
**deriva %**, **escala GT/est**, y para secuencias de **ida y vuelta**
(loop) las métricas honestas de **cierre de lazo** (`loop_closure_m`,
`max_excursion_m`): el ATE along-track es ciego al colapso de un loop.

## 4. Setup (dejarlo listo para correr)

Requisitos: Linux con GPU NVIDIA (CUDA), Python 3.8–3.10.

```bash
# 1. Dependencias Python (instala la rueda torch +cuXXX acorde a tu CUDA)
pip install -r requirements.txt

# 2. Fork base de DPVO + NUESTRA modificación
git clone https://github.com/MAC-VO/S_DPVO.git third_party/S_DPVO
cd third_party/S_DPVO && git checkout f7266f7
git apply ../../dpvo_metric/dpvo_metric.patch     # nuestra modificación (escala métrica)
cp ../../dpvo_metric/config/*.yaml config/         # nuestros configs
pip install . --no-build-isolation                 # compila extensiones CUDA
cd ../..   # (ver el README del fork para deps de build: lietorch, etc.)

# 3. Pesos DPVO (14 MB) -> third_party/S_DPVO/weights/dpvo.pth
#    https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip  (contiene dpvo.pth)
#    sha256(dpvo.pth)=30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12
mkdir -p third_party/S_DPVO/weights   # descomprime dpvo.pth aquí

# 4. (opcional) ZED SDK 5.x de Stereolabs -> trae pyzed. Solo para el baseline
#    ZED PT (run_zed_pt.py) y la conversión SVO->PNG. No se instala por pip.

# 5. Datos: coloca tus secuencias en data/recordings/ (los configs/runs/*.yaml
#    apuntan ahí, relativos). Convierte un SVO a pares estéreo con:
python scripts/svo_to_stereo_pngs.py --svo tu_video.svo \
    --out data/recordings/mi_seq --scale 0.75
```

> El repo **no incluye** el código upstream (DPVO/MAC-VO), los pesos ni los
> datos (pesados / con licencia propia): los pasos de arriba los obtienen.
> Atribución en [`NOTICE.md`](NOTICE.md).

## 5. Reproducir

```bash
# a) GT TUM desde los timecodes de cintas
python eval/build_gt_tum.py --config configs/tape_timecodes.yaml --out results/gt/

# b) DPVO métrico (nuestra modificación)
python scripts/run_sdpvo_metric.py --config configs/runs/zed2i_gym_video1.yaml \
    --inject prior_insolver --prior-strength 1000

# c) Evaluar contra el GT (ATE along-track + cierre de lazo)
python eval/eval_ate_tape_gt.py \
    --gt results/gt/gym_video_1_gt_tum.txt \
    --est results/<run_dir>/trajectory.txt --json out.json
```

## 6. Resultados (resumen)

| Secuencia | DPVO métrico (ATE) | MAC-VO | ZED PT |
|---|---|---|---|
| gym_v1 (aire, ~2 m malla) | **0.114 m** | 0.258 | 1.161 |
| gym_v2 (aire, ~1 m malla) | **0.257 m** | 1.771 | 2.664 |
| gym_v3 (aire, ~0.5 m malla) | **0.274 m** | 1.449 | 1.991 |
| video_4 (agua, loop) | 2.382 m | 3.007 | 2.660 |

**En aire** nuestra modificación supera a los métodos del estado del arte
evaluados; la ventaja crece a corta distancia de la malla (aliasing del
patrón repetitivo). **Bajo el agua**, en el loop, los tres colapsan
(cierre de lazo 3.8–4.9 m sobre un recorrido que debía cerrar en 0).

## 7. Trabajo futuro (Hito 3)

Fusión inercial con filtro de Kalman (recuperar el giro del loop) y
restricciones de plano como anclaje geométrico frente al patrón repetitivo
de la malla. Detalle en la sección final del informe.

# Odometría Visual para Mallas de Acuicultura: Hitos 2 y 3 (IPD441)

Entrega de código de los Hitos 2 y 3: evaluación comparativa de algoritmos
de odometría visual (VO) sobre mallas de acuicultura con una cámara
estéreo ZED 2i, en aire y bajo el agua (Hito 2), y la mejora del Hito 3:
**fusión inercial tight-coupling** con la IMU de la misma cámara, que
ancla el rumbo al giroscopio dentro del bundle adjustment (sección 7).

**Autores:** Ernesto Gamero, Fernanda Quintana, UTFSM.
**Informe:** el PDF escrito (formato IEEE-conf) se entrega por separado; este repositorio contiene el código de reproducción y evaluación.

## Estructura

```
dpvo_metric/   NUESTRA modificación de DPVO (escala métrica + factor IMU): patch + src/ + configs
scripts/       corrida de los modelos, extracción de IMU del SVO, demo on/off (.rrd / .mp4)
eval/          marco de evaluación (ATE along-track + cierre de lazo + rumbo vs giroscopio)
configs/       ground truth (cintas y giros) + calibración + configs por secuencia
NOTICE.md      atribución de terceros (DPVO MIT, MAC-VO, ZED SDK)
```

---

## 1. Qué se evalúa

Cinco configuraciones sobre la misma cámara y los mismos datos (la
quinta, DPVO métrico + IMU, es la mejora del Hito 3, sección 7):

| Modelo | Qué es | Escala | fps (x86 RTX 3060) |
|---|---|---|---|
| **DPVO mono** | DPVO monocular [Teed 2023], punto de partida | arbitraria | ~27 |
| **DPVO métrico** | **nuestra modificación**: prior de profundidad estéreo *in-solver* | métrica | ~19 (6.1 embarcado) |
| **DPVO métrico + IMU** | **mejora Hito 3**: factor inercial *tight-coupling* en el BA | métrica | ~15 |
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
En el Hito 3 el mismo solver se extendió con un **factor inercial de
preintegración** (sección 7). **El código de la modificación está en
[`dpvo_metric/`](dpvo_metric/)**: el patch (`dpvo_metric.patch`, 1360
líneas sobre `MAC-VO/S_DPVO@f7266f7`), los archivos modificados completos
(`src/`) y los configs. Ver [`dpvo_metric/README.md`](dpvo_metric/README.md)
para dónde está cada cambio y la matemática de ambos factores.

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
# a) GT TUM desde los timecodes de cintas (configs/gt/tape_timecodes.yaml)
python eval/build_gt_tum.py --out-dir results/gt/

# b) DPVO métrico (nuestra modificación) sobre un SVO
python scripts/run_sdpvo_metric.py --svo data/recordings/<video>.svo2 \
    --calib configs/calib/<calib>.txt \
    --config-sdpvo third_party/S_DPVO/config/sweep_p24_ow3_lt6.yaml \
    --inject prior_insolver --prior-strength 1000

# c) Evaluar contra el GT (ATE along-track + cierre de lazo)
python eval/eval_ate_tape_gt.py \
    --gt results/gt/gym_video_1_gt_tum.txt \
    --est results/<run_dir>/trajectory.txt --json out.json
```

## 6. Resultados (resumen)

### ATE: error de posición (alineamiento rígido, sin escala)

DPVO métrico se reporta como la **corrida de ATE mediano de N=3** (no la mejor),
igual que el informe; el ruido del GT (~±0.2 m por cinta) vuelve no
significativas las diferencias < 0.1 m.

| Secuencia | DPVO métrico (ATE, mediana N=3) | MAC-VO | ZED PT |
|---|---|---|---|
| gym_v1 (aire, ~2 m malla) | **0.135 m** | 0.258 | 1.161 |
| gym_v2 (aire, ~1 m malla) | **0.257 m** | 1.771 | 2.664 |
| gym_v3 (aire, ~0.5 m malla) | **0.330 m** | 1.449 | 1.991 |
| video_4 (agua, loop) | 2.382 m\* | 3.007\* | 2.660\* |

> **El ATE centrado subestima el colapso, también en aire.** El ATE alinea
> cada serie solo en *offset* (resta la media; convención `evo`/Umeyama), así
> que a un MAC-VO/ZED PT colapsado en gym_v2/v3 lo deja *flotando* en la mitad
> del GT y su ATE (1.77/1.45 m MAC-VO; 2.66/1.99 m ZED PT) **parece moderado**.
> La magnitud real del fallo es la **deriva**: MAC-VO 74.9 % / 73.8 % y ZED PT
> 104 % / 86.8 % en gym_v2/v3 (≈ varios metros de error neto), frente a la
> deriva ≤ 5.3 % de DPVO métrico. Es el **mismo fenómeno que en el loop** (ver
> abajo): para trayectorias colapsadas pero cortas, leer el ATE **junto a la
> deriva**, no de forma aislada.

\* **En un *loop* (ida y vuelta) el ATE along-track es ciego al colapso**:
centra y proyecta a 1-D, y la deriva se diluye porque el desplazamiento
neto debería ser ≈0. La cifra de aspecto moderado (2.382 m) **no comunica**
que la trayectoria no regresa. La métrica honesta es el **error de cierre
de lazo**:

### Cierre de lazo en video_4 (la métrica honesta del loop)

| Modelo | Cierre ‖fin−inicio‖ | Excursión máx / GT | Path recorrido | Escala GT/est |
|---|---|---|---|---|
| Referencia (GT) | 0.0 m | 9.0 / 9.0 m | 18.0 m | 1.000 |
| **DPVO métrico** | **4.9 m** | 4.9 / 9.0 m | 50.2 m | 0.783 |
| MAC-VO | **3.8 m** | 3.8 / 9.0 m | 22.7 m | 0.281 |
| DPVO mono | 54.5 m | 55.8 / 9.0 m | 280.6 m | 0.092 |

El GT cierra el lazo (vuelve a la cinta 1, cierre ≈ 0), pero ambos métodos
**avanzan ~la mitad del recorrido y no regresan**: el "ida y vuelta" se
pierde por completo. Expresado como deriva honesta (cierre / recorrido de
18 m) es **27 %** (DPVO métrico) y **21 %** (MAC-VO), no el engañoso
16.9 / 15.2 % que daría la fórmula estándar sobre un loop.

### Recorrido (path length) por video: DPVO métrico crudo

Cifras del Cuadro III del informe (corrida de ATE mediano de N=3). GT = camino
de referencia (en `video_4` el loop recorre ~9 m de ida + ~9 m de vuelta = 18 m).

| Secuencia | Path estimado | GT | Path / GT |
|---|---|---|---|
| gym_v1 (aire) | 14.2 m | 8.9 m | 1.6× |
| gym_v2 (aire) | 17.4 m | 9.0 m | 1.9× |
| gym_v3 (aire) | 18.4 m | 8.0 m | 2.3× |
| **video_4 (agua, loop)** | **50.2 m** | 18.0 m | **2.8×** |

El exceso sobre el GT es **temblor de alta frecuencia** (un filtro Savitzky–Golay
lo reduce sin cambiar el ATE). En aire es modesto (1.6–2.3×); en `video_4` el
recorrido se dispara a **50 m sobre un loop de 18 m** (path/GT 2.8×, pero **~10×
el desplazamiento neto** de 4.9 m): es el **peor caso del benchmark** y la firma
del colapso subacuático.

**En aire** nuestra modificación supera a los métodos del estado del arte
evaluados; la ventaja crece a corta distancia de la malla (aliasing del
patrón repetitivo). **Bajo el agua**, en el loop, los tres colapsan.

## 7. Mejora Hito 3: fusión inercial tight-coupling (demo IMU on/off)

La debilidad central del Hito 2 era el **rumbo (heading)**: en secuencias
con giro, la VO pura sobre-rota sobre la malla repetitiva y el recorrido
se deforma. La mejora integra la **IMU de la ZED 2i (~406 Hz)** como
**factor de preintegración (Forster et al. 2017) dentro del bundle
adjustment** (tight coupling): el giroscopio ancla la rotación entre
keyframes y el prior de profundidad sigue anclando la escala. Detalle de
la implementación (jacobiano analítico, regularización de velocidad,
desacople rumbo/escala con `--imu-sig-a`) en
[`dpvo_metric/README.md`](dpvo_metric/README.md).

### Reproducir el demo on/off

Dataset: secuencias **en aire** con giro deliberado de ~130° (recorrido
en V, cintas cada 1 m; GT en `configs/gt/gym_2026-06-30_giro_timecodes.yaml`).
En nuestros SVO subacuáticos el giroscopio viene muerto (todo ceros), por
eso el demo IMU es en aire por diseño.

```bash
# 1) extraer la IMU del SVO2 (imprime un gate de validez: tasa ~406 Hz y |g|~9.8)
python scripts/extract_imu_svo.py data/recordings/<video>.svo2 --out results/imu/<seq>

# 2) correr ON (con IMU) y OFF (VO pura, mismo comando sin los flags --imu*)
python scripts/run_sdpvo_metric.py --svo data/recordings/<video>.svo2 \
    --calib configs/calib/zed2i_gym_2026-06-30_hd720.txt \
    --config-sdpvo third_party/S_DPVO/config/sweep_p24_ow3_lt6.yaml \
    --inject prior_insolver --prior-strength 1000 --depth-mode NEURAL \
    --stride 1 --skip 15 --scale 0.5 --smooth-window 9 \
    --imu results/imu/<seq> --imu-strength 10 --imu-sig-a 15

# 3) métrica líder: rumbo de la cámara vs giroscopio (gyro = verdad física)
python eval/eval_heading_vs_gyro_gt.py

# 4) demo visual: .rrd interactivo (Rerun) y .mp4 offline
python scripts/onoff_to_rerun.py       # requiere rerun-sdk
python scripts/onoff_demo_video.py     # requiere ffmpeg

# (barrido de --imu-sig-a que fijó el punto de operación)
bash scripts/run_imu_siga_sweep.sh && python eval/eval_imu_siga_sweep.py
```

### Resultado (N=3, secuencias gym con giro)

Métricas del demo, por secuencia. **Rumbo**: pico de heading en el giro,
en grados; el giroscopio integrado es la referencia física, independiente
de la visión, y `|Δ|` es el error de pico de ON contra él. **Escala**:
mediana de la escala estimada/GT por brazos del recorrido (cintas cada
1 m). **fps**: efectivo end-to-end en x86 RTX 3060 (HD720@60, escala 0.5,
stride 1, 4400 a 5200 frames por secuencia). ON = tight coupling con el
punto de operación `--imu-strength 10 --imu-sig-a 15`; OFF = VO pura
métrica (mismo comando sin `--imu*`).

| Secuencia | Gyro (pico) | **IMU ON: \|Δ\| vs gyro** | IMU OFF (pico) | Escala ON | fps ON | fps OFF |
|---|---|---|---|---|---|---|
| v1 | 138° | **1°** | -212° | 0.93 | 15.3 | 22.0 |
| v2 | 127° | **0°** | 220° | 0.87 | 15.3 | 22.2 |
| v3 | -144° | **0°** | 251° | 0.99 | 15.1 | 22.3 |

**ON sigue al giroscopio dentro de ~1° y vuelve a ~0 al cerrar el
recorrido; OFF sobre-rota 210 a 251 grados y deriva.** Con el desacople
`--imu-sig-a 15` la escala queda además near-metric (0.93 / 0.87 / 0.99),
así que la planta ON reproduce la forma en V del recorrido real. El
factor IMU cuesta ~31% de fps (22 a 15, unos 20 ms extra por frame en
p50) gracias al jacobiano de pose analítico; con `--imu-v-reg 10`
(default) los 9/9 runs corren sin divergencia.

### Resultados negativos (documentados por honestidad)

Antes de la IMU se evaluaron sistemáticamente **anclajes geométricos de
plano** (coplanaridad in-solver, plano local por ventana, restricción
vertical por gravedad) y un **front-end de triangulación estéreo
dispersa**: todos negativos para esta geometría (la malla repetitiva
aliasea el matching y la escena no es coplanar). El código queda en el
patch con default OFF (flags `--plane-*` y `--stereo-*`) como registro
del avance; el análisis está en el informe.

## 8. Trabajo futuro

El colapso subacuático restante es un problema de **flujo óptico** (tasa
de refresco baja frente a la velocidad aparente de la malla), no del
solver: la escalera siguiente es hardware (cámara global-shutter de mayor
fps, brújula, sensor de profundidad) y grabar secuencias subacuáticas
con giroscopio vivo para extender el demo IMU al agua.

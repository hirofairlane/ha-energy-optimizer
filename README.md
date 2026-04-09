# Energy Optimizer — Add-on para Home Assistant OS

Gestión energética inteligente con reglas de tarifa + modelo ML (scikit-learn).  
Controla batería solar, aerotermia y depuradora de piscina.

---

## Instalación

### Método A — Local (desarrollo / uso personal)

> No necesita GitHub. Funciona directamente desde el sistema de archivos de HAOS.

1. **Accede a los archivos de HA** mediante el add-on **Samba** o **SSH + Terminal**

2. **Copia la carpeta** `ha-energy-optimizer/` a:
   ```
   /addons/ha-energy-optimizer/
   ```
   La estructura debe quedar así:
   ```
   /addons/
   └── ha-energy-optimizer/
       ├── config.yaml
       ├── Dockerfile
       ├── build.yaml
       ├── repository.yaml
       └── rootfs/
           └── usr/bin/
               ├── run.sh
               └── energy_optimizer.py
   ```

3. En HA: **Configuración → Add-ons → Tienda de complementos**  
   Menú ⋮ (esquina superior derecha) → **"Comprobar actualizaciones"**

4. Aparecerá una nueva sección **"Add-ons locales"** con el add-on listado → **Instalar**

5. En la pestaña **Configuración** del add-on:  
   - Revisa/ajusta los entity_id si los tuyos difieren de los valores por defecto  
   - El token de la API **no es necesario configurarlo manualmente** — HAOS lo inyecta automáticamente via `SUPERVISOR_TOKEN`

6. **Iniciar** el add-on y activar **"Mostrar en barra lateral"**

---

### Método B — Repositorio GitHub (actualizaciones fáciles)

> Permite actualizar el add-on desde la UI de HA sin acceso a archivos.

1. Sube **todo el contenido** de esta carpeta a un repositorio GitHub público  
   (el `repository.yaml` debe estar en la **raíz** del repo)

2. Edita `repository.yaml` con tu usuario y URL reales

3. En HA: **Configuración → Add-ons → Tienda → ⋮ → Repositorios**  
   Pega la URL HTTPS de tu repo de GitHub → **Añadir**

4. El add-on aparecerá en la tienda → **Instalar** → **Iniciar**

---

## Configuración de parámetros

Todos los entity_id están pre-configurados con tus sensores reales.  
Puedes ajustar desde la pestaña **Configuración** del add-on:

| Parámetro | Descripción | Defecto |
|-----------|-------------|---------|
| `decision_interval_minutes` | Frecuencia del ciclo de decisión | `15` |
| `retrain_cron` | Cuándo re-entrenar el modelo ML | `0 3 * * *` (3:00h) |
| `summer_start_month` / `summer_end_month` | Meses de temporada verano | `6` / `9` |
| `solar_tomorrow_irrisoria_kwh` | Umbral de producción solar "baja" para mañana | `2.0` kWh |
| `battery_emergency_threshold` | SOC% de emergencia (carga incluso en punta) | `10` |
| `battery_low_threshold` | SOC% bajo (carga en valle) | `30` |
| `battery_medium_threshold` | SOC% medio (carga moderada en valle) | `50` |

---

## Lógica de decisión

### Tarifa eléctrica
| Franja | Precio | Cuándo |
|--------|--------|--------|
| Valle  | 0.08 €/kWh | 00:00–08:00 h + sábado/domingo completo |
| Punta  | 0.30 €/kWh | 10:00–14:00 h y 18:00–22:00 h (lun–vie) |
| Llano  | 0.18 €/kWh | Resto de horas laborables |
| Venta  | 0.06 €/kWh | Venta de excedentes (fijo siempre) |

### Batería
- **En punta**: nunca cargar de red (salvo SOC < 10% → emergencia)
- **En valle**: cargar si SOC < 30% o si producción mañana < 2 kWh y SOC < 80%
- **En llano**: solo actuar si SOC < 20%

### Aerotermia (setpoint — la máquina gestiona la inercia térmica)
**Verano** (`number.ebusd_ctls2_z1coolingtemp_tempv`):
- SOC ≥ 99% cargando con placas → **16°C**
- Salón > 26°C y es de día → **20°C**
- Base → **25°C**

**Invierno** (`number.ebusd_ctls2_z1manualtemp_tempv`):
- SOC ≥ 99% cargando con placas → **18.5°C**
- Salón < 16°C y es de día → **17°C**
- Base → **16°C**

### Depuradora (`switch.depuradora`)
- Jun–Sep: 1h/día — preferencia excedente solar > valle > llano
- Oct–May: 1h/semana — misma preferencia de precio/solar

---

## Modelo ML

Entrena un **GradientBoostingRegressor** con el historial del SOC de batería  
extraído de la API REST de HA (`/api/history`) — compatible con el recorder nativo de HAOS.

Features usadas: hora del día, día de la semana, mes, SOC lag-1h, lag-4h, media móvil 4h, proxy solar.  
Se re-entrena automáticamente cada noche a las 03:00 (configurable).  
Si no hay suficiente historial, el sistema opera solo con reglas fijas sin degradación funcional.

---

## API interna

Accesible desde el panel web (Ingress) o directamente:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Panel web |
| `/api/status` | GET | Sensores + precio + info del modelo |
| `/api/decisions?limit=N` | GET | Últimas N decisiones |
| `/api/run` | POST | Ejecutar ciclo de decisión ahora |
| `/api/retrain` | POST | Re-entrenar modelo ML |
| `/api/options` | GET | Configuración activa (sin tokens) |

---

## Datos persistentes (`/data/`)

| Fichero | Contenido |
|---------|-----------|
| `options.json` | Configuración inyectada por el Supervisor de HAOS |
| `model.pkl` | Modelo ML entrenado (pipeline scikit-learn) |
| `decisions.json` | Log de últimas 500 decisiones tomadas |

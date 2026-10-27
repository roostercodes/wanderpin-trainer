# Wanderpin scanner trainer

Every hour GitHub runs `train.py` on its own servers for about 45 minutes. It:

1. pulls fresh street photos from **Mapillary** (CC BY-SA) and license-plate photos from **Wikimedia Commons**, focusing on the countries with the fewest examples so far,
2. cuts out poles, signs, guardrails, street lamps, road posts and plates (plates are also blurred, like on Street View),
3. turns each one into numbers with Wanderpin's image model and adds them to a growing collection (kept on GitHub between runs, backed up daily to Supabase Storage `wp-models/train/`),
4. retrains the country-style classifiers and the "is it really there?" checks on everything collected,
5. saves them to the `wp_scan_heads` table, which every player's clue scanner reads.

Only numbers are kept, never the photos. Results of each run are on the run's page under **Actions**; `stats.json` is updated once a day.

## Setup (once)

Add three secrets under **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `MAPILLARY_TOKEN` | your Mapillary client token (starts with `MLY|`) |
| `SUPABASE_URL` | your project URL, like `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API Keys → the **secret** key (`sb_secret_…`). The older `service_role` key (`eyJ…`) also works. |

Then open **Actions → Train the clue scanner → Run workflow** to start the first run. After that it runs every hour.

Needs: the newest `wanderpin-setup.sql` run in Supabase, and the newest `wanderpin-vision.onnx` in Storage → `wp-models`.

Photo credits: Mapillary contributors (CC BY-SA 4.0), Wikimedia Commons contributors. City data: GeoNames (CC BY 4.0). Borders: Natural Earth.

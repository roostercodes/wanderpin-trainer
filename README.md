# Wanderpin scanner trainer

Every night GitHub runs `train.py` on its own servers. It:

1. pulls fresh street photos from **Mapillary** (CC BY-SA) and license-plate photos from **Wikimedia Commons**, focusing on the countries with the fewest examples so far,
2. cuts out poles, signs, guardrails, street lamps, road posts and plates (plates are also blurred, like on Street View),
3. turns each one into numbers with Wanderpin's image model and adds them to a growing collection in Supabase Storage (`wp-models/train/`),
4. retrains the country-style classifiers and the "is it really there?" checks on everything collected,
5. saves them to the `wp_scan_heads` table, which every player's clue scanner reads.

Only numbers are kept, never the photos. Results of each run are in `stats.json` and on the run's page under **Actions**.

## Setup (once)

Add three secrets under **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `MAPILLARY_TOKEN` | your Mapillary client token (starts with `MLY|`) |
| `SUPABASE_URL` | your project URL, like `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API Keys → `service_role` / secret key |

Then open **Actions → Train the clue scanner → Run workflow** to start the first run. After that it runs every night.

Needs: the newest `wanderpin-setup.sql` run in Supabase, and the newest `wanderpin-vision.onnx` in Storage → `wp-models`.

Photo credits: Mapillary contributors (CC BY-SA 4.0), Wikimedia Commons contributors. City data: GeoNames (CC BY 4.0). Borders: Natural Earth.

# Wanderpin scanner trainer

Every hour GitHub runs `train.py` on its own servers for about 45 minutes. It:

1. finds street photos on **Mapillary**'s coverage map (CC BY-SA) and license-plate photos on **Wikimedia Commons**, focusing on the countries with the fewest examples so far. 16 downloads run side by side, and each map square is fetched once per run and reused,
2. cuts out the whole scene (4 views from 360° photos), plus poles, signs, guardrails, street lamps, road posts and plates. Each cut-out also gets a second, shifted and zoomed framing, and plates get a blurred copy, like on Street View,
3. turns each cut-out into numbers with Wanderpin's image model and adds it to a growing collection, with a small copy of the cut-out for fine-tuning. The collection stays on GitHub between runs (`cache/collection.db`), and its numbers are backed up daily to Supabase Storage `wp-models/train2/`,
4. retrains everything on the whole collection:
   - the country-style classifiers,
   - the **area** classifier (which part of the country, from where the photos were taken),
   - the "is it really there?" checks.

   For each one it tries a straight-line classifier and a small neural network, and keeps whichever is right more often on places it has never seen,
5. saves them to the `wp_scan_heads` table, which every player's clue scanner reads.

**Once a week**, when at least 30,000 cut-outs with pictures have been collected, a run fine-tunes the image model's last 2 layers on the collection. It does this instead of collecting, and it takes a few hours. The new layers are published (`wp-models/branch-vN.onnx`, about 7 MB) only if they beat the current ones on unseen places; players then download just that small file.

**The bigger image model** (`big.py`): a test of bigger models on our own photos (`bench.py`, Oct 30) showed CLIP ViT-B/16 with DataComp weights clearly ahead of TinyCLIP for telling countries apart. The trainer makes an int8 copy of it once (about 89 MB), uploads it to Supabase Storage in 2 parts (`wp-models/wanderpin-big-1.onnx.part0/1`, under the 50 MB file limit) and lets it read the collection's cut-outs a little every run (whole runs at first, until half are read). Each country/area style is then trained on the plain, tuned and bigger-model numbers, and keeps whichever is best on unseen places; a style that picks the bigger model also carries the best of the others, for devices that don't run it. Players download it only once a style uses it. Turn it off with the variable `USE_BIG=0`.

`models/wanderpin-vision-2.onnx` is the players' image model. The trainer uploads it to Supabase Storage when it's missing or changed. `models/branch_init.npz` holds the original weights of the 2 layers that get fine-tuned.

Results of each run are on the run's page under **Actions**; `stats.json` is updated once a day.

## Setup (once)

Add three secrets under **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `MAPILLARY_TOKEN` | your Mapillary client token (starts with `MLY|`) |
| `SUPABASE_URL` | your project URL, like `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API Keys → the **secret** key (`sb_secret_…`). The older `service_role` key (`eyJ…`) also works. |

Then open **Actions → Train the clue scanner → Run workflow** to start the first run. After that it runs every hour.

Needs: the newest `wanderpin-setup.sql` run in Supabase.

Photo credits: Mapillary contributors (CC BY-SA 4.0), Wikimedia Commons contributors. City data: GeoNames (CC BY 4.0). Borders: Natural Earth.

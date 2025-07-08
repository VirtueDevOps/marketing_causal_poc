#!/usr/bin/env bash
set -e
echo "→ Generating data"
python data_generator.py
echo "→ Building ontology"
python build_ontology.py
echo "→ Running causal POC"
python main.py --data data/marketing_data.zip --ontology ontology/marketing_ontology.owl
echo "✅ Done. See causal_results.png"

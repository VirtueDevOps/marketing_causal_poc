import os
from owlready2 import get_ontology, Thing, ObjectProperty

ONTO_DIR = "ontology"
ONTO_PATH = os.path.join(ONTO_DIR, "marketing_ontology.owl")

# Create a fresh ontology (no external imports)
onto = get_ontology(f"file://{os.path.abspath(ONTO_PATH)}")

with onto:
    # Domain classes under default Thing
    class Campaign(Thing): pass
    class Impression(Thing): pass
    class Click(Thing): pass
    class Conversion(Thing): pass

    # A generic 'causes' property
    class causes(ObjectProperty): pass

    # Link classes according to our funnel
    Campaign.causes   = [Impression]
    Impression.causes = [Click]
    Click.causes      = [Conversion]

# Ensure the folder exists, then save
os.makedirs(ONTO_DIR, exist_ok=True)
onto.save(file=ONTO_PATH, format="rdfxml")
print(f"Ontology written to {ONTO_PATH}")

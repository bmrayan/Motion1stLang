#!/usr/bin/env python
import argparse
import random
from pathlib import Path

from _motion_utils import ensure_dir, save_json, set_seed


NAMES = ["Alex", "Emma", "Noah", "Mia", "Liam", "Zoe", "Ivy", "Owen", "Nina", "Kai"]
COLORS = ["red", "blue", "green", "yellow", "orange", "white"]
ANIMALS = ["cat", "dog", "rabbit", "fox", "bear", "horse"]
LOCATIONS = ["kitchen", "bedroom", "garden", "hallway", "garage", "bathroom"]
OBJECTS = ["box", "ball", "bone", "cookie", "cup", "book", "crate", "toy"]


def rotate_options(correct_label, options, variant_idx):
    labels = ["A", "B", "C", "D"][: len(options)]
    correct_index = next(idx for idx, option in enumerate(options) if option == correct_label)
    target_index = variant_idx % len(options)
    reordered = list(options)
    reordered[target_index], reordered[correct_index] = reordered[correct_index], reordered[target_index]
    return [{"label": labels[idx], "text": option} for idx, option in enumerate(reordered)], labels[target_index]


def stage1_templates(rng, variant_idx):
    a, b = rng.sample(COLORS, 2)
    animal = rng.choice(ANIMALS)
    templates = [
        (
            f"A {a} ball and a {b} ball are on a table. The {a} ball rolls behind a box. "
            f"The {b} ball stays still. An object emerges from behind the box. "
            "What color is the object that emerged?",
            [a, b, "green", "cannot determine"],
            a,
        ),
        (
            f"Three {animal}s enter a room. Two {animal}s walk behind a curtain. One {animal} stays visible. "
            "How many animals are in the room?",
            ["1", "2", "3", "4"],
            "3",
        ),
        (
            f"A truck is carrying a {rng.choice(OBJECTS)}. The truck drives into a tunnel. "
            "When a vehicle exits the tunnel, what is it carrying?",
            ["nothing", "a box", "a ball", "cannot determine"],
            "a box" if "box" in "a box" else "a box",
        ),
        (
            "A ball is rolling east at constant speed. It disappears behind a wall on the east side of the room. "
            "Where will it appear next?",
            ["west side", "east side", "north side", "it will not appear"],
            "east side",
        ),
    ]
    prompt, options, correct = templates[variant_idx % len(templates)]
    return prompt, options, correct


def build_probes_for_stage(stage_name, stage_index, template_builder, rng):
    probes = []
    for template_idx in range(4):
        for variant in range(50):
            prompt, options, correct = template_builder(rng, template_idx)
            rotated, answer = rotate_options(correct, options, variant)
            probes.append(
                {
                    "id": f"{stage_name}_{template_idx}_{variant}",
                    "stage": stage_index,
                    "stage_name": stage_name,
                    "prompt": prompt,
                    "options": rotated,
                    "answer": answer,
                }
            )
    return probes


def build_stage1(rng):
    def builder(local_rng, template_idx):
        colors = local_rng.sample(COLORS, 2)
        animal = local_rng.choice(ANIMALS)
        carried = local_rng.choice(OBJECTS)
        templates = [
            (
                f"A {colors[0]} ball and a {colors[1]} ball are on a table. The {colors[0]} ball rolls behind a box. "
                f"The {colors[1]} ball stays still. An object emerges from behind the box. What color is it?",
                [colors[0], colors[1], local_rng.choice(COLORS), "cannot determine"],
                colors[0],
            ),
            (
                f"Three {animal}s enter a room. Two {animal}s walk behind a curtain. One {animal} stays visible. "
                "How many animals are in the room?",
                ["1", "2", "3", "4"],
                "3",
            ),
            (
                f"A truck is carrying a {carried}. The truck drives into a tunnel. "
                "When it exits, what is it carrying?",
                ["nothing", f"a {carried}", f"a {local_rng.choice(OBJECTS)}", "cannot determine"],
                f"a {carried}",
            ),
            (
                "A ball is rolling east at constant speed. It disappears behind a wall on the east side of the room. "
                "Where will it appear next?",
                ["west side", "east side", "north side", "it will not appear"],
                "east side",
            ),
        ]
        return templates[template_idx]

    return build_probes_for_stage("object_tracking", 1, builder, rng)


def build_stage2(rng):
    def builder(local_rng, template_idx):
        person = local_rng.choice(NAMES)
        target = local_rng.choice(LOCATIONS)
        detour_item = local_rng.choice(OBJECTS)
        templates = [
            (
                f"{person} walks past the kitchen, past the bathroom, and enters the {target}. "
                f"What was {person} most likely heading toward?",
                ["kitchen", "bathroom", target, "garden"],
                target,
            ),
            (
                f"A dog is running toward a {detour_item}. A fence blocks the direct path. "
                f"The dog runs left around the fence and continues toward the {detour_item}. Is the path efficient?",
                [
                    "yes, it took the shortest available route",
                    "no, it should have stopped",
                    "no, it went the wrong direction",
                    "cannot determine",
                ],
                "yes, it took the shortest available route",
            ),
            (
                f"{person} keeps turning toward the {target} and avoids other rooms. "
                f"What goal is {person} most likely pursuing?",
                [target, "wandering randomly", "leaving the building", "hiding"],
                target,
            ),
            (
                f"A bird flies around a tree and then lands beside a nest. "
                "What was the bird most likely trying to reach?",
                ["the nest", "the sky", "the river", "nothing in particular"],
                "the nest",
            ),
        ]
        return templates[template_idx]

    return build_probes_for_stage("goal_attribution", 2, builder, rng)


def build_stage3(rng):
    def builder(local_rng, template_idx):
        hitter = local_rng.choice(["ball A", "cart", "robot", "child"])
        target = local_rng.choice(["ball B", "vase", "cup", "stack of books"])
        templates = [
            (
                f"{hitter} rolls toward stationary {target}. {hitter} hits {target}. {target} starts moving. "
                f"What caused {target} to move?",
                [f"{hitter} hitting it", "gravity", "wind", "it moved on its own"],
                f"{hitter} hitting it",
            ),
            (
                "A cup is on the edge of a table. A cat bumps the table. The cup falls. "
                "If the cat had not bumped the table, what would have happened to the cup?",
                ["it would still fall", "it would stay on the table", "it would move left", "it would float"],
                "it would stay on the table",
            ),
            (
                "A lamp turns on after someone presses a switch. What is the best explanation?",
                ["pressing the switch caused it", "the lamp guessed", "nothing caused it", "the floor moved it"],
                "pressing the switch caused it",
            ),
            (
                "A toy car stops after hitting a wall. Why did it stop?",
                ["the wall blocked its motion", "it wanted to rest", "the room changed color", "it forgot to move"],
                "the wall blocked its motion",
            ),
        ]
        return templates[template_idx]

    return build_probes_for_stage("causal_reasoning", 3, builder, rng)


def build_stage4(rng):
    def builder(local_rng, template_idx):
        hider = local_rng.choice(NAMES)
        mover = local_rng.choice([name for name in NAMES if name != hider])
        item = local_rng.choice(["cookie", "toy", "book", "key"])
        box_a, box_b = local_rng.sample(["red box", "blue box", "green box", "yellow box"], 2)
        templates = [
            (
                f"{hider} puts a {item} in the {box_a} and leaves. While {hider} is gone, {mover} moves the {item} to the {box_b}. "
                f"When {hider} comes back, where will {hider} look first?",
                [box_a, box_b, "on the floor", "nowhere"],
                box_a,
            ),
            (
                f"{hider} sees a ball in the closet and leaves the room. {mover} moves it to the drawer. "
                f"Where does {hider} think the ball is?",
                ["in the closet", "in the drawer", "outside", "under the bed"],
                "in the closet",
            ),
            (
                f"{hider} watches a toy go into the cupboard. Then {hider} closes their eyes and {mover} moves it to a basket. "
                f"Where will {hider} expect the toy to be?",
                ["in the cupboard", "in the basket", "in the hallway", "cannot determine"],
                "in the cupboard",
            ),
            (
                f"{hider} leaves a note on the desk and exits. {mover} puts it in a drawer. "
                f"Where will {hider} first search for the note?",
                ["on the desk", "in the drawer", "in the trash", "in a box"],
                "on the desk",
            ),
        ]
        return templates[template_idx]

    return build_probes_for_stage("belief_modeling", 4, builder, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cognitive benchmark probes.")
    parser.add_argument("--output_path", type=str, default="/workspace/data/cognitive_benchmarks.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    probes = build_stage1(rng) + build_stage2(rng) + build_stage3(rng) + build_stage4(rng)

    ensure_dir(str(Path(args.output_path).parent))
    save_json(args.output_path, {"num_probes": len(probes), "probes": probes})
    print(f"Saved {len(probes)} probes to {args.output_path}")


if __name__ == "__main__":
    main()

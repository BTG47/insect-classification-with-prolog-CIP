% ----------------------------------------------
% Reglas multiclase para la capa simbólica
% ----------------------------------------------

valid_case(I) :-
    cnn_insect(I, _, P),
    P >= 0.50.

visual_support(I) :-
    part_seen(I, insect).

signature_mosquito(I) :-
    trait(I, wing_count, one_pair, P1),
    P1 >= 0.70,
    trait(I, forewing_type, membranous, P2),
    P2 >= 0.70,
    trait(I, mouthpart_type, piercing_sucking, P3),
    P3 >= 0.70,
    trait(I, body_shape, slender, P4),
    P4 >= 0.70.

signature_bee(I) :-
    trait(I, wing_count, two_pairs, P1),
    P1 >= 0.70,
    trait(I, forewing_type, membranous, P2),
    P2 >= 0.70,
    trait(I, mouthpart_type, chewing, P3),
    P3 >= 0.70,
    trait(I, waist_shape, narrow_waist, P4),
    P4 >= 0.70.

signature_grasshopper(I) :-
    trait(I, wing_count, two_pairs, P1),
    P1 >= 0.70,
    trait(I, forewing_type, tegmina, P2),
    P2 >= 0.70,
    trait(I, mouthpart_type, chewing, P3),
    P3 >= 0.70,
    trait(I, leg_specialization, jumping, P4),
    P4 >= 0.70.

signature_mantis(I) :-
    trait(I, wing_count, two_pairs, P1),
    P1 >= 0.70,
    trait(I, forewing_type, tegmina, P2),
    P2 >= 0.70,
    trait(I, mouthpart_type, chewing, P3),
    P3 >= 0.70,
    trait(I, leg_specialization, grasping, P4),
    P4 >= 0.70.

signature_butterfly(I) :-
    trait(I, wing_count, two_pairs, P1),
    P1 >= 0.70,
    trait(I, forewing_type, scaly, P2),
    P2 >= 0.70,
    trait(I, mouthpart_type, siphoning, P3),
    P3 >= 0.70,
    trait(I, antenna_type, clavate, P4),
    P4 >= 0.70.

signature_lady_beetle(I) :-
    trait(I, wing_count, two_pairs, P1),
    P1 >= 0.70,
    trait(I, forewing_type, elytra, P2),
    P2 >= 0.70,
    trait(I, body_shape, rounded_domed, P3),
    P3 >= 0.70.

final_class(I, mosquito) :-
    valid_case(I),
    visual_support(I),
    signature_mosquito(I).

final_class(I, bee) :-
    valid_case(I),
    visual_support(I),
    signature_bee(I).

final_class(I, grasshopper) :-
    valid_case(I),
    visual_support(I),
    signature_grasshopper(I).

final_class(I, mantis) :-
    valid_case(I),
    visual_support(I),
    signature_mantis(I).

final_class(I, butterfly) :-
    valid_case(I),
    visual_support(I),
    signature_butterfly(I).

final_class(I, lady_beetle) :-
    valid_case(I),
    visual_support(I),
    signature_lady_beetle(I).

final_species(I, asian_tiger_mosquito) :-
    final_class(I, mosquito),
    cnn_insect(I, asian_tiger_mosquito, P),
    P >= 0.60.

final_species(I, honey_bee) :-
    final_class(I, bee),
    cnn_insect(I, honey_bee, P),
    P >= 0.60.

final_species(I, carolina_grasshopper) :-
    final_class(I, grasshopper),
    cnn_insect(I, carolina_grasshopper, P),
    P >= 0.60.

final_species(I, european_mantid) :-
    final_class(I, mantis),
    cnn_insect(I, european_mantid, P),
    P >= 0.60.

final_species(I, monarch_butterfly) :-
    final_class(I, butterfly),
    cnn_insect(I, monarch_butterfly, P),
    P >= 0.60.

final_species(I, sevenspotted_lady_beetle) :-
    final_class(I, lady_beetle),
    cnn_insect(I, sevenspotted_lady_beetle, P),
    P >= 0.60.

needs_review(I) :-
    \+ final_class(I, _).
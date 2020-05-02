from .fields import BITWIDTH_CHOICES, Int, Bool, Str, CStr, Array, ObjArray, ftypes, filetypes
from .structbase import Struct


class Star(Struct):
    type = None
    encrypted = False

    dx = Int(10)
    y = Int(12)
    name_id = Int(10)

    def adjust(self):
        self.file.stars -= 1


class Type0(Struct):
    """ End of file """
    type = 0
    encrypted = False

    info = Int(option=filetypes('hst', 'xy', 'm'))


class Type1(Struct):
    """ Waypoint 0 Orders (1-Byte) """
    type = 1

    object_lhs = Int()
    object_rhs = Int()
    type_lhs = Int(4)
    type_rhs = Int(4)
    cargo_bits = Int(8) # ir, bo, ge, pop, fuel
    # Note: the following are evaluated as signed ints
    cargo = Array(head=None, bitwidth=8,
                  length=lambda s: bin(s.cargo_bits).count('1'))


class Type2(Struct):
    """ Waypoint 0 Orders (2-Byte) """
    type = 2

    object_lhs = Int()
    object_rhs = Int()
    type_lhs = Int(4)
    type_rhs = Int(4)
    cargo_bits = Int(8) # ir, bo, ge, pop, fuel
    # Note: the following are evaluated as signed ints
    cargo = Array(head=None, bitwidth=16,
                  length=lambda s: bin(s.cargo_bits).count('1'))


class Type3(Struct):
    """ Delete Waypoint """
    type = 3

    fleet_id = Int(11)
    unknown1 = Int(5)
    sequence = Int(8)
    unknown2 = Int(8)


class Type4(Struct):
    """ Add Waypoint """
    type = 4

    fleet_id = Int()
    sequence = Int()
    x = Int()
    y = Int()
    object_id = Int()
    order = Int(4)
    warp = Int(4)
    intercept_type = Int(8)
    transport = Array(bitwidth=8, head=None, length=None)


class Type5(Struct):
    """ Modify Waypoint """
    type = 5

    fleet_id = Int()
    sequence = Int()
    x = Int()
    y = Int()
    object_id = Int()
    order = Int(4)
    warp = Int(4)
    intercept_type = Int(8)
    transport = Array(bitwidth=8, head=None, length=None)


def type6_trigger(S):
    return S.optional_section

class Type6(Struct):
    """ Race data """
    type = 6

    player = Int(8)
    num_ship_designs = Int(8)
    planets_known = Int()
    visible_fleets = Int(12)
    station_designs = Int(4)
    unknown1 = Int(2, value=3)
    optional_section = Bool()
    race_icon = Int(5)
    unknown2 = Int(8) # 227 is computer control
    # optional section
    unknown3 = Int(32, option=type6_trigger) # not const
    password_hash = Int(32, option=type6_trigger)
    mid_G = Int(8, option=type6_trigger)
    mid_T = Int(8, option=type6_trigger)
    mid_R = Int(8, option=type6_trigger)
    min_G = Int(8, option=type6_trigger)
    min_T = Int(8, option=type6_trigger)
    min_R = Int(8, option=type6_trigger)
    max_G = Int(8, option=type6_trigger)
    max_T = Int(8, option=type6_trigger)
    max_R = Int(8, option=type6_trigger)
    growth = Int(8, option=type6_trigger)
    cur_energy = Int(8, option=type6_trigger)
    cur_weapons = Int(8, option=type6_trigger)
    cur_propulsion = Int(8, option=type6_trigger)
    cur_construction = Int(8, option=type6_trigger)
    cur_electronics = Int(8, option=type6_trigger)
    cur_biotech = Int(8, option=type6_trigger)
    # no idea yet
    whatever = Array(bitwidth=8, length=30, option=type6_trigger)
    col_per_res = Int(8, option=type6_trigger)
    res_per_10f = Int(8, option=type6_trigger)
    f_build_res = Int(8, option=type6_trigger)
    f_per_10kcol = Int(8, option=type6_trigger)
    min_per_10m = Int(8, option=type6_trigger)
    m_build_res = Int(8, option=type6_trigger)
    m_per_10kcol = Int(8, option=type6_trigger)
    leftover = Int(8, option=type6_trigger)
    energy = Int(8, max=2, option=type6_trigger)
    weapons = Int(8, max=2, option=type6_trigger)
    propulsion = Int(8, max=2, option=type6_trigger)
    construction = Int(8, max=2, option=type6_trigger)
    electronics = Int(8, max=2, option=type6_trigger)
    biotech = Int(8, max=2, option=type6_trigger)
    prt = Int(option=type6_trigger)
    imp_fuel_eff = Bool(option=type6_trigger)
    tot_terraform = Bool(option=type6_trigger)
    adv_remote_mine = Bool(option=type6_trigger)
    imp_starbases = Bool(option=type6_trigger)
    gen_research = Bool(option=type6_trigger)
    ult_recycling = Bool(option=type6_trigger)
    min_alchemy = Bool(option=type6_trigger)
    no_ramscoops = Bool(option=type6_trigger)
    cheap_engines = Bool(option=type6_trigger)
    only_basic_mine = Bool(option=type6_trigger)
    no_adv_scanners = Bool(option=type6_trigger)
    low_start_pop = Bool(option=type6_trigger)
    bleeding_edge = Bool(option=type6_trigger)
    regen_shields = Bool(option=type6_trigger)
    ignore = Int(2, value=0, option=type6_trigger)
    unknown4 = Int(8, value=0, option=type6_trigger)
    f1 = Bool(option=type6_trigger)
    f2 = Bool(option=type6_trigger)
    f3 = Bool(option=type6_trigger)
    f4 = Bool(option=type6_trigger)
    f5 = Bool(option=type6_trigger)
    p75_higher_tech = Bool(option=type6_trigger)
    f7 = Bool(option=type6_trigger)
    f_1kTlessGe = Bool(option=type6_trigger)
    # no idea yet
    whatever2 = Array(bitwidth=8, length=30, option=type6_trigger)
    unknown5 = Array(8, option=type6_trigger)
    # end optional section
    race_name = CStr(8)
    plural_race_name = CStr(8)


class Type7(Struct):
    """ Game definition """
    type = 7

    game_id = Int(32)
    size = Int()
    density = Int()
    num_players = Int()
    num_stars = Int()
    start_distance = Int()
    unknown1 = Int()
    flags1 = Int(8)
    unknown2 = Int(24)
    req_pct_planets_owned = Int(8)
    req_tech_level = Int(8)
    req_tech_num_fields = Int(8)
    req_exceeds_score = Int(8)
    req_pct_exceeds_2nd = Int(8)
    req_exceeds_prod = Int(8)
    req_capships = Int(8)
    req_highscore_year = Int(8)
    req_num_criteria = Int(8)
    year_declared = Int(8)
    unknown3 = Int()
    game_name = Str(length=32)

    def adjust(self):
        self.file.stars = self.num_stars


class Type8(Struct):
    """ Beginning of file """
    type = 8
    encrypted = False

    magic = Str(length=4, value="J3J3")
    game_id = Int(32)
    file_ver = Int()
    turn = Int()
    player = Int(5)
    salt = Int(11)
    filetype = Int(8)
    submitted = Bool()
    in_use = Bool()
    multi_turn = Bool()
    gameover = Bool()
    shareware = Bool()
    unused = Int(3)

    def __init__(self, sfile):
        super(Type8, self).__init__(sfile)
        sfile.counts.clear()

    def adjust(self):
        self.file.prng_init(self.game_id, self.turn, self.player,
                            self.salt, self.shareware)
        self.file.type = ftypes[self.filetype]


# class Type12(Struct):
#     """ Internal Messages """
#     type = 12

#     messages = ObjArray(head=None, length=None, bitwidth=)


class Type13(Struct):
    """ Authoritative Planet """
    type = 13

    planet_id = Int(11, max=998)
    player = Int(5)
    low_info = Bool(value=True) # add station design, if relevant
    med_info = Bool(value=True) # add minerals & hab
    full_info = Bool(value=True) # add real pop & structures
    const = Int(4, value=0)
    homeworld = Bool()
    f0 = Bool(value=True)
    station = Bool()
    terraformed = Bool()
    facilities = Bool() # turns on 8 bytes; rename
    artifact = Bool()
    surface_min = Bool()
    routing = Bool() # turns on 2 bytes
    f7 = Bool()
    s1 = Int(2, max=1, choices=BITWIDTH_CHOICES)
    s2 = Int(2, max=1, choices=BITWIDTH_CHOICES)
    s3 = Int(4, max=1, choices=BITWIDTH_CHOICES)
    frac_ir_conc = Int('s1')
    frac_bo_conc = Int('s2')
    frac_ge_conc = Int('s3')
    ir_conc = Int(8)
    bo_conc = Int(8)
    ge_conc = Int(8)
    grav = Int(8)
    temp = Int(8)
    rad = Int(8)
    grav_orig = Int(8, option=lambda s: s.terraformed)
    temp_orig = Int(8, option=lambda s: s.terraformed)
    rad_orig = Int(8, option=lambda s: s.terraformed)
    apparent_pop = Int(12, option=lambda s: s.player < 16) # times 400
    apparent_defense = Int(4, option=lambda s: s.player < 16)
    s4 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.surface_min)
    s5 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.surface_min)
    s6 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.surface_min)
    s7 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.surface_min)
    ir_surf = Int('s4')
    bo_surf = Int('s5')
    ge_surf = Int('s6')
    population = Int('s7') # times 100
    frac_population = Int(8, max=99, option=lambda s: s.facilities)
    mines = Int(12, option=lambda s: s.facilities)
    factories = Int(12, option=lambda s: s.facilities)
    defenses = Int(8, option=lambda s: s.facilities)
    unknown3 = Int(24, option=lambda s: s.facilities)
    station_design = Int(4, max=9, option=lambda s: s.station)
    station_flags = Int(28, option=lambda s: s.station)
    routing_dest = Int(option=lambda s: s.routing and s.player < 16)


class Type14(Struct):
    """ Scanned Planet """
    type = 14

    planet_id = Int(11, max=998)
    player = Int(5)
    # collapse these 4 fields into a single info_level field
    low_info = Bool() # add station design, if relevant
    med_info = Bool() # add minerals & hab
    full_info = Bool() # add real pop & structures
    const = Int(4, value=0)
    # /collapse
    homeworld = Bool()
    f0 = Bool(value=True)
    station = Bool()
    terraformed = Bool()
    facilities = Bool(value=False) # turns on 8 bytes; rename
    artifact = Bool()
    surface_min = Bool()
    routing = Bool() # turns on 2 bytes
    f7 = Bool()
    s1 = Int(2, max=1, choices=BITWIDTH_CHOICES,
             option=lambda s: s.med_info or s.full_info)
    s2 = Int(2, max=1, choices=BITWIDTH_CHOICES,
             option=lambda s: s.med_info or s.full_info)
    s3 = Int(4, max=1, choices=BITWIDTH_CHOICES,
             option=lambda s: s.med_info or s.full_info)
    frac_ir_conc = Int('s1')
    frac_bo_conc = Int('s2')
    frac_ge_conc = Int('s3')
    ir_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    bo_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    ge_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    grav = Int(8, option=lambda s: s.med_info or s.full_info)
    temp = Int(8, option=lambda s: s.med_info or s.full_info)
    rad = Int(8, option=lambda s: s.med_info or s.full_info)
    grav_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    temp_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    rad_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    apparent_pop = Int(12, option=lambda s: # times 400
                           (s.med_info or s.full_info) and s.player < 16)
    apparent_defense = Int(4, option=lambda s:
                               (s.med_info or s.full_info) and s.player < 16)
    s4 = Int(2, choices=BITWIDTH_CHOICES,
             option=lambda s: s.full_info and s.surface_min)
    s5 = Int(2, choices=BITWIDTH_CHOICES,
             option=lambda s: s.full_info and s.surface_min)
    s6 = Int(2, choices=BITWIDTH_CHOICES,
             option=lambda s: s.full_info and s.surface_min)
    s7 = Int(2, choices=BITWIDTH_CHOICES,
             option=lambda s: s.full_info and s.surface_min)
    ir_surf = Int('s4')
    bo_surf = Int('s5')
    ge_surf = Int('s6')
    station_design = Int(8, max=9, option=lambda s: s.station)
    last_scanned = Int(option=filetypes('h'))


class Type16(Struct):
    """ Authoritative Fleet """
    type = 16

    fleet_id = Int(9)
    player = Int(7)
    player2 = Int()
    info_level = Int(8)
    flags = Int(8)
    planet_id = Int()
    x = Int()
    y = Int()
    design_bits = Int()
    count_array = Array(bitwidth=lambda s: 16 - (s.flags & 0x8),
                        length=lambda s: bin(s.design_bits).count('1'))
    s1 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s2 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s3 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s4 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s5 = Int(8, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    ironium = Int('s1', option=lambda s: s.info_level >= 4)
    boranium = Int('s2', option=lambda s: s.info_level >= 4)
    germanium = Int('s3', option=lambda s: s.info_level >= 4)
    colonists = Int('s4', option=lambda s: s.info_level >= 7)
    fuel = Int('s5', option=lambda s: s.info_level >= 7)
    dmg_design_bits = Int()
    damage_amts = ObjArray(bitwidth=(('pct_of_type_damaged', 7),
                                     ('damage', 9)),
                           length=lambda s: bin(s.dmg_design_bits).count('1'))
    battle_plan = Int(8)
    queue_len = Int(8)


class Type17(Struct):
    """ Alien Fleet """
    type = 17

    fleet_id = Int(9)
    player = Int(7)
    player2 = Int()
    info_level = Int(8)
    flags = Int(8)
    planet_id = Int()
    x = Int()
    y = Int()
    design_bits = Int()
    count_array = Array(bitwidth=lambda s: 16 - (s.flags & 0x8),
                        length=lambda s: bin(s.design_bits).count('1'))
    s1 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s2 = Int(2, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    s3 = Int(12, choices=BITWIDTH_CHOICES, option=lambda s: s.info_level >= 4)
    ironium = Int('s1', option=lambda s: s.info_level >= 4)
    boranium = Int('s2', option=lambda s: s.info_level >= 4)
    germanium = Int('s3', option=lambda s: s.info_level >= 4)
    dx = Int(8)
    dy = Int(8)
    warp = Int(4)
    unknown2 = Int(12)
    mass = Int(32)


class Type19(Struct):
    """ Orders-at Waypoint """
    type = 19

    x = Int()
    y = Int()
    planet_id = Int()
    order = Int(4)
    warp = Int(4)
    intercept_type = Int(8)
    ir_quant = Int(12)
    ir_order = Int(4)
    bo_quant = Int(12)
    bo_order = Int(4)
    ge_quant = Int(12)
    ge_order = Int(4)
    col_quant = Int(12)
    col_order = Int(4)
    fuel_quant = Int(12)
    fuel_order = Int(4)


class Type20(Struct):
    """ Waypoint """
    type = 20

    x = Int()
    y = Int()
    planet_id = Int()
    order = Int(4)
    warp = Int(4)
    intercept_type = Int(8)


class Type21(Struct):
    """ Fleet Name """
    type = 21

    name = CStr()


class Type23(Struct):
    """ Split Fleet """
    type = 23

    fleet_id = Int(11)
    unknown = Int(5)
    fleet2_id = Int(11)
    unknown2 = Int(5)
    thirtyfour = Int(8, value=34)
    design_bits = Int()
    # Note: the following are interpreted as negative numbers
    adjustment = Array(head=None, bitwidth=16,
                       length=lambda s: bin(s.design_bits).count('1'))


class Type24(Struct):
    """ Original Fleet on Split """
    type = 24

    fleet_id = Int(11)
    unknown = Int(5)


class Type26(Struct):
    """ Ship & Starbase Design """
    type = 26

    info_level = Int(8)
    unknown = Array(bitwidth=8, length=5)
    slots_length = Int(8, option=lambda s: s.info_level > 3)
    initial_turn = Int(option=lambda s: s.info_level > 3)
    total_constructed = Int(32, option=lambda s: s.info_level > 3)
    current_quantity = Int(32, option=lambda s: s.info_level > 3)
    slots = ObjArray(bitwidth=(('flags', 16),
                               ('part_sub_id', 8),
                               ('quantity', 8)),
                     length='slots_length',
                     option=lambda s: s.info_level > 3)
    name = CStr()


class Type27(Struct):
    """ New Ship & Starbase Design """
    type = 27

    info_level = Int(4)
    player_id = Int(4)
    index = Int(8)
    unknown = Array(bitwidth=8, length=6, option=lambda s: s.info_level)
    slots_length = Int(8, option=lambda s: s.info_level)
    initial_turn = Int(option=lambda s: s.info_level)
    total_constructed = Int(32, value=0, option=lambda s: s.info_level)
    current_quantity = Int(32, value=0, option=lambda s: s.info_level)
    slots = ObjArray(bitwidth=(('flags', 16),
                               ('part_sub_id', 8),
                               ('quantity', 8)),
                     length='slots_length',
                     option=lambda s: s.info_level)
    name = CStr(option=lambda s: s.info_level)


class Type28(Struct):
    """ New Turn Queue State """
    type = 28

    queue = ObjArray(length=None, head=None, bitwidth=(('quantity', 10),
                                                       ('build_type', 6),
                                                       ('unknown', 8),
                                                       ('frac_complete', 8)))


class Type29(Struct):
    """ Update Queue """
    type = 29

    planet_id = Int()
    queue = ObjArray(length=None, head=None, bitwidth=(('quantity', 10),
                                                       ('build_type', 6),
                                                       ('unknown', 8),
                                                       ('frac_complete', 8)))


class Type30(Struct):
    """ Battle plans """
    type = 30

    id = Int(8)
    flags = Int(8)
    u1 = Int(4, option=lambda s: not (s.flags & 64))
    u2 = Int(4, option=lambda s: not (s.flags & 64))
    u3 = Int(4, option=lambda s: not (s.flags & 64))
    u4 = Int(4, option=lambda s: not (s.flags & 64))
    name = CStr(option=lambda s: not (s.flags & 64))


class Type37(Struct):
    """ Merge Fleet """
    type = 37

    fleets = Array(bitwidth=16, head=None, length=None)


class Type40(Struct):
    """ In-game messages """
    type = 40

    unknown1 = Int(32)
    sender = Int()
    receiver = Int()
    unknown2 = Int()
    text = CStr(16)


class Type43(Struct):
    """ Minefields / Debris / Mass Packets / Wormholes / Mystery Trader """
    type = 43

    # optional sizes: 2, 4, and 18
    quantity = Int(option=lambda s: s.file.type in ('hst', 'm') and
                   s._vars._seq == 0)

    index = Int(9, option=lambda s: s.quantity is None)
    owner = Int(4, option=lambda s: s.quantity is None)
    misc_type = Int(3, max=3, option=lambda s: s.quantity is None)
    detonate = Int(option=filetypes('x'))
    x = Int(option=lambda s: s.quantity is None and s.detonate is None)
    y = Int(option=lambda s: s.quantity is None and s.detonate is None)

    # minefields
    num_mines = Int(option=lambda s: s.detonate is None and s.misc_type == 0)
    zero1 = Int(value=0, option=lambda s: s.detonate is None
                and s.misc_type == 0)
    flags_mf = Array(length=6, option=lambda s: s.detonate is None
                     and s.misc_type == 0)
    # end minefields

    # debris / mass packets
    dest_planet_id = Int(10, option=lambda s: s.detonate is None
                         and s.misc_type == 1)
    unknown_mp = Int(6, option=lambda s: s.detonate is None
                     and s.misc_type == 1)
    mass_ir = Int(option=lambda s: s.detonate is None and s.misc_type == 1)
    mass_bo = Int(option=lambda s: s.detonate is None and s.misc_type == 1)
    mass_ge = Int(option=lambda s: s.detonate is None and s.misc_type == 1)
    flags_mp = Int(option=lambda s: s.detonate is None and s.misc_type == 1)
    # end debris / mass packets

    # wormholes
    flags_wh = Array(length=10, option=lambda s: s.detonate is None
                     and s.misc_type == 2)
    # end wormholes

    # mystery trader
    x_end = Int(option=lambda s: s.detonate is None and s.misc_type == 3)
    y_end = Int(option=lambda s: s.detonate is None and s.misc_type == 3)
    warp = Int(4, option=lambda s: s.detonate is None and s.misc_type == 3)
    unknown_mt1 = Int(12, value=1, option=lambda s: s.detonate is None
                      and s.misc_type == 3)
    interceptions = Int(option=lambda s: s.detonate is None
                        and s.misc_type == 3)
    unknown_mt2 = Int(option=lambda s: s.detonate is None and s.misc_type == 3)
    # end mystery trader

    previous_turn = Int(option=lambda s: s.quantity is None and
                        s.detonate is None)


class Type45(Struct):
    """ Score data """
    type = 45

    player = Int(5)
    unknown1 = Bool(value=True) # rare False?
    f_owns_planets = Bool()
    f_attains_tech = Bool()
    f_exceeds_score = Bool()
    f_exceeds_2nd = Bool()
    f_production = Bool()
    f_cap_ships = Bool()
    f_high_score = Bool()
    unknown2 = Bool(value=False) # rare True?
    f_declared_winner = Bool()
    unknown3 = Bool()
    year = Int() # or rank in .m files
    score = Int(32)
    resources = Int(32)
    planets = Int()
    starbases = Int()
    unarmed_ships = Int()
    escort_ships = Int()
    capital_ships = Int()
    tech_levels = Int()


# class Type31(Struct):
#     """ Battle Recording """
#     type = 31

#     battle_id = Int(8)
#     unknown1 = Array(bitwidth=8, length=3)
#     participant_bits = Int()
#     total_len = Int()
#     planet_id = Int()
#     x = Int()
#     y = Int()


# class Type38(Struct):
#     """ Player Relations """
#     type = 38

#     relations = Array(length=lambda s: s.sfile.num_players)

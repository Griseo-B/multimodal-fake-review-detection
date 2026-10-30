"""Researcher-written fake-review skeletons for domains that have no crowd gold set.

Each skeleton is original text. Slot fills change the dish or product, the staff
name, and a number. Callers must keep every fill of one skeleton in the same
split, so a test skeleton is not a rewrite of a training sentence.
"""

from __future__ import annotations

FOOD_SLOTS = [
    {"item": "mushroom tagliatelle", "staff": "Maya", "n": "12"},
    {"item": "soy-glazed salmon", "staff": "Luis", "n": "18"},
    {"item": "roasted cauliflower", "staff": "Priya", "n": "9"},
    {"item": "short rib ragu", "staff": "Owen", "n": "22"},
    {"item": "citrus ceviche", "staff": "Elena", "n": "15"},
    {"item": "black garlic ramen", "staff": "Jonah", "n": "11"},
    {"item": "lamb kofta", "staff": "Amina", "n": "20"},
    {"item": "brown butter gnocchi", "staff": "Grace", "n": "14"},
    {"item": "chili crisp dumplings", "staff": "Noah", "n": "16"},
    {"item": "miso eggplant", "staff": "Chris", "n": "13"},
]

PRODUCT_SLOTS = [
    {"item": "cordless kettle", "staff": "Maya", "n": "6"},
    {"item": "linen duvet cover", "staff": "Luis", "n": "4"},
    {"item": "cast iron skillet", "staff": "Priya", "n": "9"},
    {"item": "noise-limiting headphones", "staff": "Owen", "n": "3"},
    {"item": "glass food containers", "staff": "Elena", "n": "8"},
    {"item": "wool running socks", "staff": "Jonah", "n": "5"},
    {"item": "desk lamp", "staff": "Amina", "n": "7"},
    {"item": "travel mug", "staff": "Grace", "n": "2"},
    {"item": "baking scale", "staff": "Noah", "n": "10"},
    {"item": "handheld steamer", "staff": "Chris", "n": "6"},
]

# Positive food skeletons. Human-written, deceptive by construction.
FOOD_POS = [
    "I walked in without a booking and {staff} had us seated in {n} minutes. The {item} was the dish I came for, and the portion, salt, and heat were all where I wanted them. Water was refilled before I asked. I would bring a friend here on a weeknight and order the {item} again.",
    "Our table was ready at the time on the reservation. {staff} explained the specials without rushing the table next to us. I ordered the {item} and it arrived looking like the menu photo. We stayed about {n} minutes after dessert because the room stayed comfortable. This is now the place I suggest when someone asks for a low-risk dinner.",
    "The host stand remembered the note about a window seat. {staff} checked back once, which was enough. The {item} had a clear main flavor and the side was not an afterthought. I counted {n} small plates on the menu that I would still try. I left planning the next visit rather than comparing it with anywhere else.",
    "I ate alone at the counter and never felt parked. {staff} remembered that I wanted the {item} without onion. It came out in {n} minutes and stayed hot to the last bite. The bill matched the menu prices. I would sit at that same counter again after work.",
    "We were a group of {n} and the courses still landed together. {staff} split the {item} across two plates without being asked twice. Nobody at our table sent anything back. The room was lively but I could hear the person beside me. I already told my sister to book this room for her birthday.",
    "Lunch service was the test, and it passed. {staff} had the {item} on the table while my tea was still too hot to drink, roughly {n} minutes after I ordered. I did not need sauce, salt, or a second napkin. I will use this spot when I have a meeting nearby and only one free hour.",
]

FOOD_NEG = [
    "I arrived at the booked time and waited {n} minutes before anyone spoke to me. {staff} eventually brought the {item}, which was lukewarm in the center and oversalted at the edge. I asked for it to be replaced and the second plate had the same problem. I will not use this reservation again.",
    "The room was so loud that we stopped talking. {staff} dropped the {item} and left before we could ask for water. We flagged someone down {n} times. The dish looked nothing like the description, and the check still included a full-price entree. I would not bring anyone here.",
    "Two of the {n} plates we ordered were wrong, including the {item}. {staff} apologized and then disappeared into the kitchen. The replacement came with a different side than the one on the menu. We paid and left before dessert. I do not plan to give this kitchen another evening.",
    "I could smell fryer oil from the sidewalk. Inside, {staff} told me the {item} would take {n} minutes and it took far longer. When it arrived the garnish was wilted and the protein was dry at the ends. I sent it back and did not order a substitute. This was a wasted night.",
    "Our server {staff} recited specials and then brought a different dish from the {item} I had confirmed. The table was sticky when we sat down. I asked for a wipe and it arrived after {n} minutes. The food was edible and nothing more. I will choose another block next time.",
    "I came for the {item} because the menu described it in detail. What {staff} set down was a much smaller plate, and the missing side was not mentioned. We waited {n} minutes for the bill after asking twice. I do not dispute that the staff was polite. I do dispute that the meal matched what was offered.",
]

PRODUCT_POS = [
    "I bought the {item} after comparing three listings, and {staff} from the seller answered my sizing question the same day. It arrived in {n} days with the parts listed on the page. The first week of use matched the description closely enough that I stopped looking at alternatives. I would buy this again from the same listing.",
    "Setup took me about {n} minutes because the {item} included the fasteners it promised. {staff} had already marked the box with the model I ordered, so I did not have to guess. It has been quiet, stable, and easy to clean. I have already recommended this exact listing to a coworker.",
    "The {item} replaced an older one that I could not repair. Delivery was {n} days, and {staff} sent a photo of the label when I asked. The materials feel like the product text, not like a thinner substitute. I have used it every day this week and I have no reason to return it.",
    "I was worried the {item} would be smaller than the photos. {staff} confirmed the measurements before it shipped, and the tape measure agreed when I opened it {n} days later. Nothing was cracked. It does the job I bought it for, and I would order the same model for the office.",
    "My order of the {item} showed up a day inside the {n}-day window. The instructions were short and accurate. {staff} included the extra piece the listing mentioned, which is usually where these orders go wrong. I have already put it into daily use and I would repurchase it when this one wears out.",
    "I needed the {item} for a trip and it was on the porch in {n} days. {staff} had upgraded nothing and left nothing out. The weight, the color, and the attachments all matched the page. I used it the morning it arrived. This is the listing I will reuse.",
]

PRODUCT_NEG = [
    "The {item} arrived {n} days late and missing a part that the photo shows attached. {staff} told me to reorder instead of sending the missing piece. The unit I did receive wobbles on a flat counter. I packed it back up the same night. I will not buy this listing again.",
    "I ordered the {item} in the color named on the page. {staff} shipped a different finish and the box took {n} days. The return label was not in the package, so I had to ask twice. Even before the color issue, the hinge felt loose. I do not recommend this seller.",
    "The listing for the {item} promised a manual and a spare seal. Neither was in the box. {staff} answered once and then stopped. I waited {n} days for a follow-up that never came. The product itself leaks at the seam. I am sending it back.",
    "I assembled the {item} and one fastener was stripped out of the bag. {staff} said the hardware kit was complete. It took me {n} extra trips to a shop to finish a job the product was supposed to finish. The frame also sits lower than the stated height. I would not buy it again.",
    "The {item} smelled like solvent when I opened it, and the smell was still there after {n} days by a window. {staff} suggested I air it out rather than replace it. The power button sticks. This does not match the condition described in the listing, and I have stopped using it.",
    "What I received was not the {item} in the product photos. {staff} asked me to accept a partial refund of {n} dollars and keep it. The substitute is lighter and the cord is shorter than the one I ordered. I declined the partial refund. I do not want a replacement from this listing either.",
]


def expand(domain: str, polarity: str, skeletons: list[str], slots: list[dict]) -> list[dict]:
    rows = []
    for index, skeleton in enumerate(skeletons):
        for slot in slots:
            rows.append(
                {
                    "domain": domain,
                    "label": "human_fake",
                    "polarity": polarity,
                    "text": skeleton.format(**slot),
                    "group_id": f"template-{domain}-{polarity}-{index}",
                    "source": "researcher_template",
                    "generator": "",
                }
            )
    return rows


def researcher_templates() -> list[dict]:
    rows = []
    rows.extend(expand("food", "positive", FOOD_POS, FOOD_SLOTS))
    rows.extend(expand("food", "negative", FOOD_NEG, FOOD_SLOTS))
    rows.extend(expand("ecommerce", "positive", PRODUCT_POS, PRODUCT_SLOTS))
    rows.extend(expand("ecommerce", "negative", PRODUCT_NEG, PRODUCT_SLOTS))
    return rows

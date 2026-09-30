"""Generate a synthetic historical-ticket dataset for training the ML models.

Each ticket is built from latent traits (frustration, repeat contact, multiple
issues, technical depth). Those traits shape both the text and the outcomes
(resolution hours, escalation), so the models must learn to recover them from
the text via the sentiment/complexity features - the same way they would on
real data. Replace data/historical_tickets.csv with a real export to retrain
on production history.

Usage:  python scripts/generate_data.py [--rows 2500] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticket_router.config import HISTORICAL_TICKETS_CSV  # noqa: E402

ISSUES: dict[str, list[tuple[list[str], list[str]]]] = {
    "Billing": [
        (["Charged twice for my subscription", "Double charge on my card", "Duplicate payment this month"],
         ["I was billed twice for my {plan} plan this month, two charges of ${amount} on my credit card.",
          "My bank statement shows the same ${amount} transaction twice on {date}.",
          "You took the subscription payment two times. Please refund the duplicate charge."]),
        (["Refund request", "Requesting a refund for order {order}", "Want my money back"],
         ["I cancelled within the trial period but was still charged ${amount}. I would like a refund.",
          "Please process a refund for order {order}, I never used the service after paying.",
          "I requested a refund two weeks ago for ${amount} and have not received it."]),
        (["Unauthorized charge", "Money taken without permission", "Unknown debit on my statement"],
         ["There is a ${amount} debit from your company on my account that I never authorized.",
          "Money was taken from my bank account without my permission, ${amount} on {date}.",
          "I see a payment to you I don't recognise. I did not approve this transaction."]),
        (["Invoice amount is wrong", "Incorrect invoice", "Invoice does not match my plan"],
         ["The invoice for {date} shows ${amount} but my plan should cost less.",
          "My latest invoice includes fees I don't recognise. Can you explain the charges?",
          "The receipt lists the {plan} plan but I'm on a cheaper plan."]),
        (["Payment declined", "Cannot update payment method", "Card keeps getting declined"],
         ["My payment keeps getting declined even though the card works everywhere else.",
          "I tried to update my credit card details but the payment page rejects the card.",
          "The subscription renewal failed and now I can't pay the outstanding balance."]),
        (["Question about pricing and discounts", "Upgrade billing question", "Cancel subscription billing"],
         ["If I upgrade to the {plan} plan mid-cycle, how will the billing be prorated?",
          "Is there a discount coupon for annual billing? I want to switch from monthly payment.",
          "I want to cancel my subscription at the end of the billing period, will I be charged again?"]),
    ],
    "Technical": [
        (["App crashes on startup", "Application keeps crashing", "Crash when uploading files"],
         ["The app crashes every time I try to upload a file larger than 10MB.",
          "Since the latest update the application crashes right after the splash screen.",
          "The mobile app freezes and then crashes when I open the reports section."]),
        (["Error 500 on dashboard", "Server error when saving", "Getting error messages"],
         ["I get a 500 internal server error whenever I open the dashboard.",
          "Saving a project fails with the message 'unexpected error, please try again'.",
          "The page shows error code {error} and nothing loads."]),
        (["Data not syncing between devices", "Sync is broken", "Changes not saving"],
         ["Changes I make on my laptop don't sync to my phone anymore.",
          "The sync has been stuck on 'pending' for hours and my data is out of date.",
          "Edits made by my team are not showing up, looks like sync failed."]),
        (["API requests timing out", "Integration not working", "Webhook failures"],
         ["Our API calls to the orders endpoint keep timing out after 30 seconds.",
          "The Slack integration stopped posting notifications yesterday.",
          "Webhooks are failing with a timeout and our integration is broken."]),
        (["Website very slow", "Pages take forever to load", "Export to CSV not working"],
         ["The website has been extremely slow, pages take over a minute to load.",
          "Exporting my data to CSV produces an empty file.",
          "The search feature is not working, it just keeps loading."]),
    ],
    "Account": [
        (["Cannot reset my password", "Password reset link not working", "Forgot password"],
         ["The password reset email never arrives in my inbox.",
          "When I click the reset link it says the token is invalid or expired.",
          "I forgot my password and the reset form says my email is not recognised."]),
        (["Locked out of my account", "Account locked", "Account suspended"],
         ["My account got locked after a few failed login attempts.",
          "I'm locked out and can't sign in to access my projects.",
          "My account was suspended without any notice and I can't log in."]),
        (["Two-factor code not received", "2FA problem", "Lost my authenticator"],
         ["I'm not receiving the two-factor verification code by SMS.",
          "I got a new phone and lost my authenticator app, now I can't pass 2FA.",
          "The verification code is always rejected as invalid."]),
        (["Change email address", "Update my username", "Add a team member"],
         ["How can I change the email address on my account profile?",
          "I need to update my username and profile details.",
          "I want to add a new team member to my account with admin permissions."]),
        (["Delete my account", "Close my account", "Deactivate profile"],
         ["Please delete my account and all associated data.",
          "I would like to deactivate my profile and close the account.",
          "How do I permanently delete my account?"]),
    ],
    "Shipping": [
        (["Package has not arrived", "Order {order} not delivered", "Where is my order?"],
         ["My order {order} was supposed to arrive {days} days ago and it's still not here.",
          "The package was marked as shipped but never arrived.",
          "I ordered {days} days ago and still have no delivery."]),
        (["Tracking number not updating", "No tracking information", "Tracking stuck"],
         ["The tracking number for order {order} hasn't updated in {days} days.",
          "The carrier website says the tracking number doesn't exist.",
          "Tracking has shown 'in transit' at the same warehouse for a week."]),
        (["Received damaged item", "Item arrived broken", "Package was damaged"],
         ["The parcel arrived damaged and the item inside is broken.",
          "My order arrived with a cracked screen, the box was crushed by the courier.",
          "Half of the items in the package were damaged during delivery."]),
        (["Wrong item delivered", "Received someone else's order", "Incorrect item in package"],
         ["I received the wrong item, I ordered a blue one and got a red one.",
          "The package contained someone else's order.",
          "The item delivered doesn't match what I ordered in order {order}."]),
        (["Change shipping address", "Need a return label", "Update delivery address"],
         ["I need to change the shipping address for order {order} before it's dispatched.",
          "Can you send me a return label so I can send the item back?",
          "Please update my delivery address, I've moved."]),
    ],
    "General": [
        (["Question about features", "Does the product support {feature}?", "How do I use {feature}?"],
         ["I'm wondering whether your product supports {feature}.",
          "How do I get started with {feature}? I couldn't find it in the help center.",
          "Could you give me some information about what's included in each plan?"]),
        (["Feedback on the new design", "Suggestion for improvement", "Feature request"],
         ["I wanted to share some feedback on the new design, it looks cleaner.",
          "A suggestion: it would be nice to have a dark mode.",
          "It would be great if you added {feature} in a future release."]),
        (["Business hours inquiry", "Partnership inquiry", "Student discount?"],
         ["What are your support hours on weekends?",
          "We're interested in a partnership with your company, who should we contact?",
          "Do you offer a student discount or nonprofit pricing?"]),
    ],
}

FILL = {
    "plan": ["Pro", "Business", "Premium", "Team", "Enterprise"],
    "amount": ["19.99", "49.00", "99.00", "12.50", "249.00", "8.99"],
    "date": ["March 3rd", "last Monday", "the 15th", "June 1st", "yesterday"],
    "order": ["#10482", "#55321", "#98012", "#77310", "#30055", "#61209"],
    "error": ["E1042", "ERR_CONN_RESET", "502", "403", "0x80070005"],
    "days": ["3", "5", "7", "10", "12"],
    "feature": ["calendar sync", "single sign-on", "offline mode", "bulk export", "custom reports"],
    "device": ["Windows 11", "macOS Sonoma", "an iPhone 14 on iOS 17", "a Pixel 8 on Android 14", "Chrome 126"],
    "version": ["4.2.1", "4.3.0", "3.9.8", "5.0.2"],
}

GREETINGS = ["Hi,", "Hello,", "Hello team,", "Good morning,", "Hi there,", "", ""]
CLOSERS = ["Thanks.", "Please help.", "Regards.", "Thank you.", "Looking forward to your reply.", ""]

POSITIVE = [
    "I love your product and it's been great so far.",
    "Your team has always been really helpful, thanks in advance!",
    "Thanks so much, I appreciate the quick help.",
    "Overall I'm very happy with the service.",
]
FRUSTRATION_MILD = [
    "This is quite frustrating.",
    "I'm a bit disappointed, to be honest.",
    "This is annoying and is slowing down my work.",
    "Unfortunately this is causing problems for me.",
]
FRUSTRATION_STRONG = [
    "This is completely unacceptable!",
    "I am extremely frustrated and angry about this.",
    "This is the worst service I've ever experienced!!",
    "Honestly this is ridiculous and a total waste of my time.",
    "I'm furious. Your product is useless right now.",
]
ESCALATION_THREATS = [
    "I want to speak to a manager immediately.",
    "If this isn't fixed I will dispute the charge with my bank.",
    "I'm considering legal action.",
    "I will cancel my subscription and switch to a competitor.",
    "Please escalate this to a supervisor.",
    "I will be leaving a review on social media about this.",
]
PRIOR_ATTEMPTS = [
    "I already tried logging out and back in, and I cleared the cache.",
    "This is the third time I've contacted you about this.",
    "I reached out last week and got no response.",
    "I have followed the steps in your help article multiple times and it still happens.",
    "I restarted and reinstalled everything, nothing works.",
    "It keeps happening again and again for weeks now.",
]
TECH_DETAILS = [
    "I'm on {device} with app version {version}.",
    "The browser console shows an exception with code {error}.",
    "The server logs show a timeout on the API endpoint.",
    "I checked the network tab and the request returns a 502 error.",
    "Our SSL certificate and DNS configuration look correct.",
]
SECOND_ISSUE = {
    "Billing": "Also, I was charged an extra fee on my last invoice.",
    "Technical": "Additionally, the app shows an error when I try to export data.",
    "Account": "On top of that, I can't log in to my account on the mobile app.",
    "Shipping": "Also, my last package arrived late and the tracking never updated.",
    "General": "Additionally, I had a question about your support hours.",
}

BASE_HOURS = {"Billing": 10, "Technical": 22, "Account": 6, "Shipping": 16, "General": 4}
PRIORITY_SPEED = {"low": 1.5, "medium": 1.0, "high": 0.75, "urgent": 0.55}
PRIORITY_ESC = {"low": -0.4, "medium": 0.0, "high": 0.5, "urgent": 1.0}
CATEGORY_ESC = {"Billing": 0.3, "Technical": 0.3, "Account": 0.0, "Shipping": 0.2, "General": -0.8}
CATEGORY_WEIGHTS = {"Billing": 0.25, "Technical": 0.28, "Account": 0.18, "Shipping": 0.17, "General": 0.12}


def fill(text: str, rng: random.Random) -> str:
    for key, values in FILL.items():
        while "{" + key + "}" in text:
            text = text.replace("{" + key + "}", rng.choice(values), 1)
    return text


def pick_priority(frustration: int, category: str, rng: random.Random) -> str:
    weights = [0.3, 0.45, 0.18, 0.07]
    if frustration >= 1:
        weights = [0.1, 0.35, 0.35, 0.2]
    if category == "General":
        weights = [0.6, 0.35, 0.05, 0.0]
    return rng.choices(["low", "medium", "high", "urgent"], weights=weights)[0]


def make_ticket(rng: random.Random) -> dict:
    category = rng.choices(list(CATEGORY_WEIGHTS), weights=list(CATEGORY_WEIGHTS.values()))[0]
    subjects, descriptions = rng.choice(ISSUES[category])

    frustration = rng.choices([-1, 0, 1, 2], weights=[0.15, 0.45, 0.25, 0.15])[0]
    if category == "General":
        frustration = min(frustration, 1)
    prior = rng.random() < (0.15 + 0.2 * max(frustration, 0))
    multi = rng.random() < 0.2
    tech_depth = rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0] if category == "Technical" else (
        rng.choices([0, 1], weights=[0.85, 0.15])[0])
    threat = frustration == 2 and rng.random() < 0.55

    parts = [rng.choice(GREETINGS), rng.choice(descriptions)]
    if tech_depth:
        parts += rng.sample(TECH_DETAILS, tech_depth)
    if prior:
        parts += rng.sample(PRIOR_ATTEMPTS, rng.choice([1, 2]))
    if multi:
        parts.append(SECOND_ISSUE[rng.choice([c for c in SECOND_ISSUE if c != category])])
    if frustration == -1:
        parts.append(rng.choice(POSITIVE))
    elif frustration == 1:
        parts.append(rng.choice(FRUSTRATION_MILD))
    elif frustration == 2:
        parts.append(rng.choice(FRUSTRATION_STRONG))
    if threat:
        parts.append(rng.choice(ESCALATION_THREATS))
    parts.append(rng.choice(CLOSERS))

    priority = pick_priority(frustration, category, rng)

    hours = BASE_HOURS[category] * PRIORITY_SPEED[priority]
    hours *= 1 + 0.7 * multi + 0.6 * prior + 0.35 * tech_depth + 0.15 * max(frustration, 0)
    hours *= math.exp(rng.gauss(0, 0.25))

    logit = (-3.0 + 1.0 * max(frustration, 0) + 0.9 * prior + 0.6 * multi + 1.2 * threat
             + 0.2 * tech_depth + PRIORITY_ESC[priority] + CATEGORY_ESC[category] + rng.gauss(0, 0.3))
    escalated = rng.random() < 1 / (1 + math.exp(-logit))

    return {
        "subject": fill(rng.choice(subjects), rng),
        "description": fill(" ".join(p for p in parts if p), rng),
        "priority": priority,
        "category": category,
        "resolution_hours": round(hours, 1),
        "escalated": int(escalated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=HISTORICAL_TICKETS_CSV)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [make_ticket(rng) for _ in range(args.rows)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    escalation_rate = sum(r["escalated"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} tickets to {args.out} (escalation rate {escalation_rate:.1%})")


if __name__ == "__main__":
    main()

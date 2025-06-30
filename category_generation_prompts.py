#!/usr/bin/env python3
"""
Category generation prompts for different types of reasoning tasks.
Contains prompts for generating examples of listing and interleaved reasoning prompts.
"""

from typing import List


# ============================================================================
# LISTING PROMPTS
# ============================================================================

LISTING_GENERATION_PROMPT = """Generate 20 diverse examples of prompts that ask for lists. These should be prompts where the answer naturally breaks down into discrete bullet points or numbered items. Each prompt should ask for a specific number of items (between 3-5 items).

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Avoid prompts that can be answered purely from general knowledge or memory. The ideal prompts should make someone think "I need to look this up" or "I should verify this information."

CRITICAL: Each item in the list should require MULTI-STEP REASONING to produce. This means:
- Combining data from multiple sources or databases
- Performing calculations or comparisons between different metrics
- Analyzing relationships between different pieces of information
- Sequential reasoning where finding one piece of information leads to needing another
- Cross-referencing different types of data (e.g., financial + geographical + temporal)
- Synthesis of information rather than simple fact lookup

The prompts should cover various domains like:
- Technology and programming
- Business and entrepreneurship  
- Science and research
- Legal and government
- Health and medicine
- Education and learning
- Travel and geography
- Finance and economics
- Environmental issues
- Social sciences
- Current events
- Practical life skills

Prioritize prompts that involve:
- Recent events or current data (2023-2024)
- Specific statistics, numbers, or rankings
- Legal requirements or regulatory details
- Technical specifications or current best practices
- Market data or financial information
- Scientific research findings
- Government policies or procedures

Format each prompt as: "List X [specific things] for [specific context/scenario]"

Examples of multi-step reasoning listing prompts:
- "List 4 different ways to get a green card in the USA, ranked by total cost including all fees, lawyer costs, and opportunity costs over the entire process duration"
- "List 5 publicly traded companies that had the biggest market cap increase in 2024, but only include those where the increase was NOT primarily due to AI/tech hype (requires analyzing both market data and business fundamentals)"
- "List 3 countries where remote workers can get the best quality of life per dollar spent, factoring in visa requirements, tax implications, cost of living, and internet infrastructure quality"
- "List 4 pharmaceutical stocks that outperformed the S&P 500 in 2024 while also having a drug in Phase 3 trials (requires cross-referencing stock performance data with clinical trial databases)"
- "List 5 US cities where a $100k salary provides the highest effective purchasing power after accounting for taxes, housing costs, and local inflation rates"
- "List 3 climate change adaptation projects globally that achieved measurable results in 2024, ranked by cost-effectiveness per unit of impact"

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# HISTORICAL/CHRONOLOGICAL PROMPTS
# ============================================================================

HISTORICAL_CHRONOLOGICAL_PROMPT = """Generate 20 diverse prompts that ask for historical or chronological progressions. These should be prompts where each item in the response builds upon or follows from the previous item in a temporal sequence.

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each item should require MULTI-STEP REASONING combining multiple sources.

Examples of good historical/chronological prompts:
- "Detail the key stages of the Roman Republic's decline into the Roman Empire, starting with the reforms of the Gracchi brothers. Each stage should logically lead to the next, culminating in the ascension of Augustus."
- "Chronicle the key events in the development of artificial intelligence, starting with Alan Turing's theoretical work. Each development should build upon previous breakthroughs."
- "Trace the evolution of the iPhone from the original 2007 model through the iPhone 15, highlighting how each generation addressed limitations of its predecessor."

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# PROCEDURAL/INSTRUCTIONAL PROMPTS
# ============================================================================

PROCEDURAL_INSTRUCTIONAL_PROMPT = """Generate 20 diverse prompts that ask for procedural or instructional steps. These should be prompts where each step must be performed in a specific order and builds upon the previous step.

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each step should require MULTI-STEP REASONING and cannot be answered from memory alone.

Examples of good procedural/instructional prompts:
- "Detail the step-by-step process of conducting a comprehensive security audit of a web application, from initial reconnaissance to final reporting. Each step should build on information gathered in previous steps."
- "Describe the process of a bill becoming a law in the United States Congress. Start with the introduction of the bill in either the House or the Senate and detail each sequential step it must pass through."
- "Outline the sequential steps for launching a startup from initial idea to first product release. Each phase should depend on the completion of the previous phase."

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# CAUSE AND EFFECT PROMPTS
# ============================================================================

CAUSE_EFFECT_PROMPT = """Generate 20 diverse prompts that ask for cause and effect progressions. These should be prompts where each item is a direct consequence of the previous item, creating a chain reaction or domino effect.

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each cause-effect relationship should require MULTI-STEP REASONING to identify and explain.

Examples of good cause and effect prompts:
- "Illustrate the domino effect of the 2008 financial crisis, starting with the subprime mortgage market. Each point in your list should represent a subsequent, directly related consequence."
- "Explain the cascading effects of climate change on ocean ecosystems, starting with rising global temperatures. Each effect should be a direct consequence of the previous environmental change."
- "Describe the progressive failure of a complex software system under increasing load, starting with the first bottleneck. Each failure should cascade from the previous system stress."

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# NARRATIVE/STORYTELLING PROMPTS
# ============================================================================

NARRATIVE_STORYTELLING_PROMPT = """Generate 20 diverse prompts that ask for narrative or storytelling progressions. These should be prompts where each element follows a narrative arc or character development sequence.

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each narrative element should require MULTI-STEP REASONING about plot structure, character development, or story analysis.

Examples of good narrative/storytelling prompts:
- "Outline the character development arc of a protagonist who transforms from villain to hero. Each stage should show how the character's actions and realizations lead to the next phase of growth."
- "Detail the narrative structure of a psychological thriller, from the establishment of normalcy to the final revelation. Each plot point should build tension and depend on previous revelations."
- "Describe the typical stages of Joseph Campbell's 'Hero's Journey.' Each stage should be presented in its narrative order, showing the progression of the hero's adventure."

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# ORDERED/PREFERENCE PROMPTS
# ============================================================================

ORDERED_PREFERENCE_PROMPT = """Generate 20 diverse prompts that ask for ordered or ranked lists. These should be prompts where items must be presented in a specific order based on some criteria (time, cost, difficulty, effectiveness, etc.).

IMPORTANT: Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each ranking should require MULTI-STEP REASONING combining multiple metrics and data sources.

Examples of good ordered/preference prompts:
- "List different ways to get a green card in the USA, ordered by the typical number of years each process takes from start to finish."
- "Rank programming languages for web development based on job market demand, from highest to lowest demand in 2024."
- "List machine learning algorithms in order of complexity for beginners to learn, from simplest to most complex, where each builds conceptually on the previous."

Generate 20 such prompts, each on a new line, numbered 1-20:"""


# ============================================================================
# TRIP PLANNING/ITINERARY PROMPTS
# ============================================================================

TRIP_PLANNING_ITINERARY_PROMPT = """Generate 20 diverse prompts that ask for detailed trip planning or itinerary generation. These should be prompts for 2-4 day trips where each day/step of the itinerary has complex substructures that require interleaved reasoning across multiple factors.

IMPORTANT: All prompts must specify trips of 2-4 days only (no longer trips). Create prompts that would realistically require research, fact-checking, or external data sources to answer accurately. Each itinerary item should require MULTI-STEP REASONING that involves:
- Cross-referencing transportation schedules, costs, and availability
- Coordinating accommodation bookings with activity schedules
- Balancing budget constraints across multiple expense categories
- Considering seasonal factors, weather patterns, and local events
- Optimizing routes based on geographical constraints and time limits
- Researching visa requirements, health precautions, and local regulations
- Integrating cultural considerations and local customs into planning

Each step should have subparts like:
- Morning/afternoon/evening activities with specific timing
- Transportation logistics between locations
- Accommodation considerations and booking requirements
- Budget breakdown and cost optimization strategies
- Alternative plans for weather or availability issues
- Local cultural events or seasonal considerations

Examples of good trip planning/itinerary prompts (2-4 days only):
- "Create a 2-day itinerary for a family of four visiting Japan during cherry blossom season, with a $3000 budget. Each day should include detailed logistics for transportation between cities, accommodation bookings that align with travel schedules, and activities that account for seasonal crowds and weather patterns."
- "Plan a 3-day solo backpacking trip through Southeast Asia, optimizing for both cost-effectiveness and cultural immersion. Each location should include visa requirements, budget hostels with good reviews, local transportation options, and activities that provide authentic cultural experiences while considering safety for solo travelers."
- "Design a 4-day business trip to three European cities with tight meeting schedules. Each day should coordinate flight connections, hotel locations near meeting venues, backup transportation options, and brief cultural activities that fit around business obligations and jet lag recovery."
- "Create a 2-day weekend getaway to a national park, focusing on hiking and photography. Include equipment needs, trail difficulty assessments, weather considerations, and backup indoor activities."

Generate 20 such prompts (all 2-4 days), each on a new line, numbered 1-20:"""


# ============================================================================
# CATEGORY DICTIONARIES
# ============================================================================

# Interleaved reasoning category prompts
INTERLEAVED_CATEGORY_PROMPTS = {
    "historical_chronological": HISTORICAL_CHRONOLOGICAL_PROMPT,
    "procedural_instructional": PROCEDURAL_INSTRUCTIONAL_PROMPT,
    "cause_effect": CAUSE_EFFECT_PROMPT,
    "narrative_storytelling": NARRATIVE_STORYTELLING_PROMPT,
    "ordered_preference": ORDERED_PREFERENCE_PROMPT,
    "trip_planning_itinerary": TRIP_PLANNING_ITINERARY_PROMPT,
}


# All available categories
ALL_CATEGORIES = {
    "listing": LISTING_GENERATION_PROMPT,
    **INTERLEAVED_CATEGORY_PROMPTS
}


def get_generation_prompt(category: str) -> str:
    """Get the generation prompt for a specific category."""
    if category in ALL_CATEGORIES:
        return ALL_CATEGORIES[category]
    else:
        raise ValueError(f"Unknown category: {category}. Available categories: {list(ALL_CATEGORIES.keys())}")


def get_available_categories() -> List[str]:
    """Get list of all available categories."""
    return list(ALL_CATEGORIES.keys())


def get_interleaved_categories() -> List[str]:
    """Get list of interleaved reasoning categories."""
    return list(INTERLEAVED_CATEGORY_PROMPTS.keys())


if __name__ == "__main__":
    print("Available categories:")
    for category in get_available_categories():
        print(f"  - {category}")
    
    print(f"\nInterleaved categories: {get_interleaved_categories()}")
    print(f"Total categories: {len(get_available_categories())}") 
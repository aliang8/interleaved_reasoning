import argparse
import pandas as pd
import sys

# PROMPTS = [
#     "Create a 2-day trip to Kyoto, Japan, during Gion Matsuri, including festival-specific activities, accommodations in traditional ryokans, and transportation logistics around city closures or roadblocks. Include cultural etiquette notes and budget for traditional meals and event tickets.",
#     "Design a 2-day trip to Thailand for a solo traveler, covering Bangkok, Chiang Mai, and a beach destination. Include budget for mix of hostels and boutique hotels, transportation schedules (flights, buses), and activities that reflect both cultural immersion and relaxation, plus safety considerations for solo travel.",
#     "Plan a 3-day trip to Tokyo, Japan, including a visit to the Tokyo Tower, a day trip to Mount Fuji, and a night at a traditional ryokan. Include cultural etiquette notes and budget for traditional meals and event tickets.",
#     "Create a 2-day trip to the Swiss Alps, including Zermatt and the Matterhorn. Include train schedules, accommodations in mountain chalets, budget for ski passes or hiking gear, and alternative indoor activities if weather is poor. Factor in altitude considerations for travelers.",
#     "Plan a 3-day trip to Paris, France, including a visit to the Eiffel Tower, a day trip to the Louvre, and a night at a traditional French restaurant. Include cultural etiquette notes and budget for traditional meals and event tickets.",
#     "Create a 2-day trip to the Maldives, including a visit to the Maldives, a day trip to the Louvre, and a night at a traditional French restaurant. Include cultural etiquette notes and budget for traditional meals and event tickets.",
#     "Create a 2-day trip to Korea, including a visit to the DMZ, a day trip to the DMZ, and a night at a traditional French restaurant. Include cultural etiquette notes and budget for traditional meals and event tickets.",
#     "Create a 2-day trip to Brazil, including a visit to the Amazon rainforest, a day trip to the Amazon rainforest, and a night at a traditional French restaurant. Include cultural etiquette notes and budget for traditional meals and event tickets.",
# ]

PROMPTS = [
    "List 4 countries with the highest renewable energy capacity per capita in 2024, factoring in both current installed capacity and population growth projections for the next 5 years.",
    "List 3 AI startups founded after 2020 that received over $100 million in funding in 2024, while also having a commercial product with measurable revenue in at least three markets.",
    "List 5 cryptocurrencies with the highest transaction throughput in Q1 2024, ranked by comparing transactions per second (TPS), energy efficiency, and real-world adoption metrics.",
    "List 4 medical treatments for Type 2 diabetes approved after 2022 that showed statistically significant improvements in blood sugar control, along with their cost per patient year and insurance coverage rates in the US.",
    "List 3 countries that reduced their carbon emissions the most between 2022 and 2024, adjusting for GDP growth, population increase, and economic activity in energy-intensive sectors.",
    "List 5 Fortune 500 companies that implemented remote work policies in 2023 and reported measurable productivity gains or cost savings, including the specific metrics used to evaluate success.",
    "List 4 universities that increased their global rankings by the most points in 2024, based on QS, Times Higher Education, and U.S. News rankings, factoring in changes in research output and international student enrollment.",
    "List 3 cities with the most severe housing affordability crises in 2024, ranked by combining median home prices, income-to-cost ratios, and the percentage of population spending over 30% of income on housing.",
    "List 5 renewable energy projects (solar, wind, hydro) commissioned in 2023 that exceeded their projected energy output, including the actual vs. projected kWh generation and grid integration challenges faced.",
    "List 4 countries that implemented new digital tax laws in 2024, including the key provisions, tax rates on digital services, and the estimated annual revenue generated based on preliminary data.",
    "List 3 pharmaceutical drugs that received FDA breakthrough therapy designation in 2024 and entered Phase 2 clinical trials, including the target disease, current trial enrollment size, and projected market potential.",
    "List 5 countries with the most restrictive data privacy laws in 2024, ranked by the number of compliance requirements, penalties for violations, and impact on international business operations.",
    "List 4 electric vehicle models with the highest real-world range per dollar spent in 2024, factoring in battery efficiency, vehicle pricing, and government incentives.",
    "List 3 global supply chain disruptions (e.g., port closures, trade restrictions) that significantly affected manufacturing in 2024, including the duration of the disruption, number of affected industries, and estimated economic impact.",
    "List 5 fintech companies that launched a successful digital banking service in 2024, measured by user acquisition rates, average account balance, and regulatory compliance status in at least two countries.",
    "List 4 scientific studies published in 2024 that challenged a major assumption in climate science, including the key findings, institutions involved, and the impact on current climate policy models.",
    "List 3 educational technologies that saw the most adoption by K-12 schools in 2024, including the percentage of schools using the platform, reported improvements in student engagement, and teacher feedback metrics.",
    "List 5 countries with the most improved cybersecurity frameworks in 2024, based on changes in legislation, number of cyber incidents reported, and international cybersecurity rankings.",
    "List 4 global economic indicators that showed the most significant change in 2024 compared to 2023, including GDP growth rates, inflation rates, and the underlying causes of the change.",
    "List 3 cities that implemented a successful congestion pricing policy in 2024, including the number of vehicles reduced, revenue generated, and public transportation usage increase post-implementation.",
]
def main():
    parser = argparse.ArgumentParser(description="Create a Parquet file for training from a single prompt.")
    parser.add_argument('--output', type=str, default='custom_prompts.parquet', help='Output Parquet file name')
    parser.add_argument('--prompt_key', type=str, default='prompt', help='Column name for the prompt')
    args = parser.parse_args()

    with open("prompts.txt", "w") as f:
        prompts = f.readlines()

    df = pd.DataFrame([{args.prompt_key: {"content": prompt, "role": "user"}} for prompt in prompts])
    df.to_parquet(args.output, index=False)
    print(f"Wrote prompt to {args.output} with column '{args.prompt_key}'")


if __name__ == "__main__":
    main()
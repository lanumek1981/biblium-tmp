# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:42:56 2024

@author: Lan.Umek
"""

import re

g1 = (
    r"(extreme poverty|poverty alleviation|poverty eradication|poverty reduction|"
    r"international poverty line|"
    r"(financial aid.*poverty|financial aid.*poor|financial aid.*north-south divide)|"
    r"financial development.*poverty|"
    r"financial empowerment|"
    r"distributional effect|distributional effects|"
    r"child labor|child labour|"
    r"development aid|"
    r"social protection|social protection system|"
    r"social protection.*access|"
    r"microfinanc\w*|micro-financ\w*|"
    r"resilience of the poor|"
    r"safety net.*(poor|vulnerable)|"
    r"economic resource.*access|economic resources.*access|"
    r"food bank|food banks)"
)

g2 = (
    r"(land tenure rights|"
    r"(smallholder.*(farm|forestry|pastoral|agriculture|fishery|food producer|food producers))|"
    r"malnourish\w*|malnutrition|undernourish\w*|undernutrition|"
    r"agricultural production|agricultural productivity|agricultural practices|agricultural management|"
    r"food production|food productivity|food security|food insecurity|"
    r"land right|land rights|land reform|land reforms|resilient agricultural practices|"
    r"(agriculture.*potassium)|"
    r"fertili.z\w*|food nutrition improvement|hidden hunger|genetically modified food|"
    r"(gmo.*food)|agroforestry practices|agroforestry management|agricultural innovation|"
    r"(food security.*genetic diversity)|"
    r"(food market.*(restriction|tariff|access|north south divide|development governance))|"
    r"food governance|food supply chain|food value chain|food commodity market)"
    r"(?!.*disease)"  # Negative lookahead to exclude rows mentioning 'disease'
)

g3 = (
    r"((human.*(health\w*|disease\w*|illness\w*|medicine\w*|mortality))|"
    r"battered child syndrome|cardiovascular disease|cardiovascular diseases|chagas|child abuse|"
    r"child neglect|child well-being index|youth well-being index|child wellbeing index|youth wellbeing index|"
    r"water-borne disease|water-borne diseases|water borne disease|water borne diseases|tropical disease|"
    r"tropical diseases|chronic respiratory disease|chronic respiratory diseases|infectious disease|"
    r"infectious diseases|sexually-transmitted disease|sexually transmitted disease|"
    r"sexually-transmitted diseases|sexually transmitted diseases|communicable disease|communicable diseases|"
    r"aids|hiv|human immunodeficiency virus|tuberculosis|malaria|hepatitis|polio\w*|vaccin\w*|cancer\w*|"
    r"diabet\w*|maternal mortality|child mortality|childbirth complications|neonatal mortality|"
    r"neo-natal mortality|premature mortality|infant mortality|quality adjusted life year|maternal health|"
    r"preventable death|preventable deaths|tobacco control|substance abuse|drug abuse|tobacco use|"
    r"alcohol use|substance addiction|drug addiction|tobacco addiction|alcoholism|suicid\w*|"
    r"postnatal depression|post-natal depression|zika virus|dengue|schistosomiasis|sleeping sickness|"
    r"ebola|mental health|mental disorder|mental illness|mental illnesses|measles|neglected disease|"
    r"neglected diseases|diarrhea|diarrhoea|cholera|dysentery|typhoid fever|traffic accident|"
    r"traffic accidents|healthy lifestyle|life expectancy|life expectancies|health policy|"
    r"(health system.*(access|accessible))|health risk|health risks|inclusive health|obesity|"
    r"social determinants of health|psychological harm|psychological wellbeing|psychological well-being|"
    r"psychological well being|public health)"
)

g4 = (
    r"((school|education|educational).*"
    r"(school attendance|school enrollment|school enrolment|inclusive education|"
    r"educational inequality|education quality|educational enrolment|educational enrollment|"
    r"adult literacy|numeracy rate|educational environment|educational access|"
    r"(development aid.*teacher training)|early childhood education|basic education|"
    r"affordable education|educational financial aid|school safety|safety in school|"
    r"(learning opportunities.*(gender disparities|empowerment))|"
    r"(learning opportunity.*(gender disparities|empowerment))|youth empowerment|"
    r"women empowerment|equal opportunities|child labour|child labor|discriminatory|"
    r"educational gap|(poverty trap.*schooling)|special education needs|inclusive education system|"
    r"(schooling.*(gender disparities|ethnic disparities|racial disparities))|"
    r"education exclusion|education dropouts|global citizenship|sustainable development education|"
    r"environmental education|education policy|educational policies|international education|"
    r"education reform|(educational reform.*developing countries)|educational governance|"
    r"(developing countries.*school effects)|education expenditure|foreign aid|"
    r"(teacher training.*developing countries)|teacher attrition)"
    r")(?!.*health literacy)"  # Exclude mentions of 'health literacy'
)

g5 = (
    r"(gender inequality|gender equality|employment equity|gender wage gap|"
    r"female labor force participation|female labour force participation|"
    r"women labor force participation|women labour force participation|womens' employment|"
    r"female employment|women's unemployment|female unemployment|"
    r"(access.*family planning services)|forced marriage|child marriage|"
    r"forced marriages|child marriages|occupational segregation|"
    r"women's empowerment|girls' empowerment|female empowerment|"
    r"female genital mutilation|female genital cutting|domestic violence|"
    r"women.*violence|girl\w*.*violence|sexual violence|"
    r"(unpaid work.*gender inequality)|(unpaid care work.*gender inequality)|"
    r"women's political participation|female political participation|"
    r"female managers|women in leadership|female leadership|intra-household allocation|"
    r"(access.*reproductive healthcare)|honour killing|honor killing|honour killings|honor killings|"
    r"antiwomen|anti-women|feminism|misogyny|female infanticide|female infanticides|"
    r"human trafficking|forced prostitution|"
    r"(equality.*(sexual rights|reproductive rights|divorce rights))|"
    r"women's rights|gender injustice|gender injustices|gender discrimination|"
    r"gender disparities|gender gap|female exploitation|household equity|"
    r"women's underrepresentation|female entrepreneurship|female ownership|"
    r"women's economic development|women's power|gender-responsive budgeting|gender quota|"
    r"(foreign aid.*women's empowerment)|gender segregation|gender-based violence|gender participation|"
    r"female politician|female leader|contraceptive behaviour|women's autonomy|"
    r"agrarian feminism|microfinance|women's livelihood|women's ownership|"
    r"female smallholder|gender mainstreaming)"
)

g6 = (
    r"((safe.*(water access|drinking water))|"
    r"(clean.*(drinking water|water source))|"
    r"(water.*(sanitation and hygiene|sanitation & hygiene|quality|resource).*(water availability|water-use efficiency|water supply|water supplies|clean water|hygienic toilet|hygienic toilets|antifouling membrane|antifouling membranes|anti-fouling membrane|anti-fouling membranes|water management|aquatic toxicology|water toxicology|aquatic ecotoxicology|water ecotoxicology))|"
    r"((freshwater|fresh water).*(water quality).*(pollutant|pollution|contamina\w*))|"
    r"(freshwater.*(water security|water shortage|(waste water.*treatment)|(wastewater.*treatment)|water conservation|water footprint|water infrastructure|water pollution|water purification|water use|water uses|sanit\w*|sewer\w*))|"
    r"(water.*(ecosystem|eco-system).*(protection of|endocrine disruptor|endocrine disruptors)(?!.*marine))|"
    r"(water.*water management.*(pollution remediation|pollutant removal))|"
    r"((groundwater|ground water|ground-water).*freshwater)|"
    r"((water pollution|water pollutant).*((waste water.*treatment)|(wastewater.*treatment)))|"
    r"(freshwater availability|fresh water availability|water scarcity|open defecation|blue water|green water|grey water|black water))"
    r"(?!.*global burden of disease study)"
)

g7 = (
    r"(energy efficiency|energy consumption|energy transition|clean energy technology|"
    r"energy equity|energy justice|energy poverty|energy policy|renewable\w*|"
    r"2000 Watt society|smart micro-grid|smart grid|smart microgrid|smart micro-grids|"
    r"smart grids|smart microgrids|smart meter|smart meters|affordable electricity|"
    r"electricity consumption|reliable electricity|clean fuel|clean cooking fuel|"
    r"fuel poverty|energiewende|life-cycle assessment|life cycle assessment|"
    r"life-cycle assessments|life cycle assessments|"
    r"(photochemistry.*renewable energy)|photovoltaic|"
    r"photocatalytic water splitting|hydrogen production|water splitting|"
    r"lithium-ion batteries|lithium-ion battery|heat network|district heat|district heating|"
    r"residential energy consumption|domestic energy consumption|energy security|"
    r"rural electrification|energy ladder|energy access|energy conservation|"
    r"low-carbon society|hybrid renewable energy system|hybrid renewable energy systems|"
    r"fuel switching|(foreign development aid.*renewable energy)|"
    r"energy governance|(official development assistance.*electricity)|"
    r"(energy development.*developing countries))"
    r"(?!.*(wireless sensor network|wireless sensor networks))"
)

g8 = (
    r"(economic growth|economic development policy|employment policy|inclusive economic growth|"
    r"sustainable growth|economic development|economic globalization|economic globalisation|"
    r"economic productivity|low-carbon economy|inclusive growth|microfinanc\w*|micro-financ\w*|"
    r"micro-credit\w*|microcredit\w*|equal income|equal wages|decent job\w*|quality job\w*|"
    r"job creation|full employment|employment protection|informal employment|precarious employment|"
    r"unemployment|precarious job\w*|microenterprise\w*|micro-enterprise\w*|small enterprise\w*|"
    r"medium enterprise\w*|small entrepreneur\w*|starting entrepreneur\w*|medium entrepreneur\w*|"
    r"social entrepreneurship|safe working environment|labor market institution\w*|"
    r"labour market institution\w*|forced labour|forced labor|child labour|child labor|"
    r"labour right\w*|labor right\w*|modern slavery|human trafficking|child soldier\w*|global jobs|"
    r"living wage|minimum wage|circular economy|inclusive economy|rural economy|"
    r"Foreign Development Investment|Aid for Trade|trade union\w*|working poor|"
    r"Not in Education, Employment, or Training|carbon offset\w*|offset project\w*|"
    r"economic diversification|material footprint|resource efficiency|"
    r"(cradle to cradle.*economy)|economic decoupling|labour market disparities|"
    r"sustainable tourism|ecotourism|community-based tourism|tourism employment|"
    r"sustainable tourism policy|financial access|financial inclusion|access to banking)"
    r"(?!.*health)"  # Exclude mentions of 'health'
)

g9 = (
    r"(industrial growth|industrial diversification|infrastructural development|"
    r"infrastructural investment|infrastructure investment|public infrastructure|resilient infrastructure|"
    r"transborder infrastructure|public infrastructures|resilient infrastructures|transborder infrastructures|"
    r"(industrial emissions.*mitigation)|industrial waste management|industrial waste treatment|traffic congestion|"
    r"microenterprise\w*|micro-enterprise\w*|small enterprise\w*|medium enterprise\w*|small entrepreneur\w*|"
    r"medium entrepreneur\w*|value chain management|"
    r"(broadband access.*developing countries)|manufacturing innovation|manufacturing investment|"
    r"sustainable transportation|accessible transportation|transportation services|inclusive transportation|"
    r"R&D investment|green product\w*|sustainable manufacturing|"
    r"(cradle to cradle.*industry)|closed loop supply chain|"
    r"(industrial.*innovation)|process innovation|product innovation|inclusive innovation)"
)

g10 = (
    r"((equality.*(economic|financial|socio-economic))|"
    r"(inequality.*(economic|financial|socio-economic))|"
    r"economic reform policy|economic reform policies|political inclusion|"
    r"social protection policy|social protection policies|"
    r"(immigration(?!.*(chemistry|disease|biodiversity)))|"
    r"(emigration(?!.*(chemistry|disease|biodiversity)))|"
    r"foreign direct investment|development gap\w*|"
    r"migrant remittance|responsible migration|migration policy\w*|"
    r"north-south divide|"
    r"(developing.*(tariffs|tariff|zero-tariff|duty-free access))|"
    r"social exclusion|economic marginali[zs]ation|income inequality|"
    r"discriminatory law\w*|discriminatory policies|discriminatory policy|"
    r"economic empowerment|economic transformation|"
    r"(global market.*empowerment))"
)

g11 = (
    r"((city|cities|human settlement\w*|urban|metropoli\w*|town\w*|municipal\w*).*(gentrification|congestion|"
    r"transportation|public transport|housing|slum\w*|sendai framework|disaster risk reduction|drr|smart city|"
    r"smart cities|resilient building\w*|sustainable building\w*|building design|buildings design|"
    r"urbani[sz]ation|zero energy building\w*|zero-energy building\w*|basic service\w*|governance|"
    r"citizen participation|collaborative planning|participatory planning|inclusiveness|cultural heritage|"
    r"natural heritage|unesco|disaster|ecological footprint|environmental footprint|waste|pollution|"
    r"pollutant\w*|waste water|recycling|circular economy|air quality|green space\w*|nature inclusive|"
    r"nature inclusive building\w*))"
)

g12 = (
    r"((environmental pollution|hazardous waste|hazardous chemical\w*|"
    r"toxic chemical\w*|chemical pollution|ozone depletion|pesticide pollution|"
    r"pesticide stress|pesticide reduction|life cycle assessment|life cycle analysis|"
    r"life cycle analyses|life-cycle analysis|life-cycle analyses|low carbon economy|"
    r"low-carbon economy|environmental footprint|material footprint|harvest efficiency|"
    r"solid waste|waste generation|corporate social responsibility|corporate sustainability|"
    r"consumer behavior\w*|consumer behaviour\w*|waste recycling|resource recycling|resource reuse|"
    r"biobased economy|zero waste|sustainability label|sustainability labelling|"
    r"global resource extraction|material flow accounting|societal metabolism|food spill|"
    r"resource spill|resource efficiency|sustainable food consumption|green consumption|"
    r"sustainable supply chain|circular economy|cradle to cradle|sustainable procurement|"
    r"sustainable tourism|fossil-fuel subsidies|fossil-fuel expenditure|"
    r"(consumption.*(resource use|spill))|"
    r"(production.*(resource use|spill)))"
    r"(?!.*(wireless sensor network\w*|wireless network\w*|wireless\w*|disease|astrophysics)))"
)


g13 = (
    r"(climate action|climate adaptation|climate change|climate capitalism|ipcc|climate effect|climate equity|"
    r"climate feedback|climate finance|climate change financing|climate forcing|climate governance|"
    r"climate impact|climate investment|climate justice|climate mitigation|climate model\w*|climate modeling|"
    r"climate modelling|climate policy|climate policies|climate risk\w*|climate services?|climate prediction\w*|"
    r"climate signal\w*|climate tipping point|climate variation\w*|ecoclimatology|eco-climatology|"
    r"Green Climate Fund|regional climate\w*|urban climate\w*|"
    r"(climate.*(adaptive management|awareness|bioeconomy|carbon|decision-making|disaster risk reduction|"
    r"environmental education|sustainable development education|energy conservation|emission\w*|extreme|"
    r"food chain\w*|framework|hazard\w*|island\w*|land use|megacit\w*|consumption|production|"
    r"small island developing states|anthropocene|atmospher\w*|clean development mechanism|glacier retreat|"
    r"warming|greenhouse|ice-ocean interaction\w*|nitrogen cycle\w*|ocean acidification|radiative forcing|"
    r"sea ice|sea level\w*|thermal expansion|unfccc|ozone))"
    r")(?!.*(drug|geomorphology))"
)

g14 = (
    r"((marine|ocean|oceans|sea|seas|coast\w*|mangrove).*(water cycle\w*|biogeochemical cycle\w*|"
    r"oceanic circulation model\w*|oceanic circulation modelling|oceanic circulation modeling|"
    r"ice-ocean|eutrophicat\w*|coral bleach\w*|coastal management|coastal habitat\w*|marine debris|"
    r"ocean acidification|(acidification.*seawater)|fishery\w*|overfishing|sustainable yield|"
    r"marine protected area\w*|marine conservation|ecotourism|community based conservation|"
    r"community-based conservation|marine land slide|marine pollution|nutrient runoff|"
    r"coastal ecotourism|destructive fishing|local fisheries|artisanal fishers|fisheries rights|"
    r"species richness|traditional ecological knowledge|small island development states|"
    r"marine quota|marine economy|marine policy))"
    r"(?!.*(paleoclimate|paleoceanography|radiocarbon|genetics|medicine|drug|engineering|aerosol))"
)

g15 = (
    r"((terrestrial|land|inland|freshwater).*(biodivers\w*|species richness|bioeconom\w*|bio-econom\w*|"
    r"biological production|deforest\w*|desertif\w*|earth system|ecological resilience|ecosystem\w*|"
    r"eco-system\w*|trophic cascade|trophic level|trophic web|threatened species|endangered species|"
    r"extinction risk\w*|poach\w*|wildlife product\w*|wildlife traffic\w*|wildlife market\w*|"
    r"wildlife trafficking|invasive species|alien species|land use\w*|land degradation|soil degradation|"
    r"LULUCF|forest\w*|land conservation|wetland\w*|mountain\w*|dryland\w*|mountainous cover|"
    r"protected area\w*|REDD|forest management|silviculture|timber harvest|illegal logging|"
    r"slash-and-burn|fire-fallow cultivation|tree cover|soil restoration|land restoration|drought|"
    r"sustainable land management|mountain vegetation|habitat restoration|Red List species|"
    r"Red List Index|extinction wave|habitat fragmentation|habitat loss|Nagoya Protocol on Access to Genetic Resources|"
    r"genetic resources|biological invasion|biodiversity-inclusive|forest stewardship council|"
    r"rainforest alliance|forest certification|forest auditing|ecotourism|"
    r"community-based conservation|community based conservation|human-wildlife conflict))"
)

g16 = (
    r"((actual innocence|false confession|armed conflict\w*|civil conflict\w*|"
    r"(war.*(conflict|warfare|democracy|Geneva Convention|treaty|peace))|"
    r"peacekeeping|(corruption.*(institution|public official|government|bribery|conflict))|"
    r"crime\w*|criminal|democratic deficit|"
    r"(democrati[sz]ation.*(institutional|conflict|decision-making|society|politics|financial aid))|"
    r"ethnic conflict\w*|exoneration|genocid\w*|homicid\w*|murder\w*|human trafficking|"
    r"criminal justice system|justice system|arbitrary justice|refugee\w*|terroris\w*|violence|"
    r"torture|effective rule of law|arms flow|transparent institution\w*|good governance|"
    r"legal identity for all|freedom of information|human rights institution\w*|"
    r"human rights activists|fundamental freedom\w*|violent conflict\w*|peaceful society|"
    r"effective institution\w*|accountable institution\w*|inclusive institution\w*|child abuse|"
    r"arbitrary detention|unsentenced detention|judicial system|criminal tribunal|inclusive societ\w*|"
    r"responsive institution\w*|fair societ\w*|legal remedy\w*|independence of judiciary|"
    r"independent judiciary|separation of powers|extremism|war crime\w*|organized crime|"
    r"illicit transfer|illicit money|arms trafficking|cybercrime|insurgence|"
    r"democratic institution\w*|political instability|"
    r"(political decision-making.*(responsive|inclusive|participatory|representative))|"
    r"Aarhus Convention|press freedom|freedom of speech))"
    r"(?!.*(disease|genetics))"
)


pairs = [
    (g1, "SDG 01"),
    (g2, "SDG 02"),
    (g3, "SDG 03"),
    (g4, "SDG 04"),
    (g5, "SDG 05"),
    (g6, "SDG 06"),
    (g7, "SDG 07"),
    (g8, "SDG 08"),
    (g9, "SDG 09"),
    (g10, "SDG 10"),
    (g11, "SDG 11"),
    (g12, "SDG 12"),
    (g13, "SDG 13"),
    (g14, "SDG 14"),
    (g15, "SDG 15"),
    (g16, "SDG 16")
]

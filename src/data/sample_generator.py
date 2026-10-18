"""
Synthetic Indian legal dataset generator.
Generates realistic Indian Supreme Court benchmark cases, queries, citations, and qrels
grounded in real landmark Indian jurisprudence for fast local testing and CI verification.
"""

import json
import os
from typing import Dict, List, Tuple
from src.data.schema import LegalCase, LegalQuery, Qrel


def generate_synthetic_legal_dataset() -> Tuple[List[LegalCase], List[LegalQuery], List[Qrel]]:
    """
    Constructs a high-fidelity synthetic benchmark corpus of landmark Indian Supreme Court judgments,
    queries representing novel legal controversies, and gold-standard relevance judgments.
    """
    cases = [
        LegalCase(
            doc_id="SC_1973_KESHAVANANDA",
            title="Kesavananda Bharati Sripadagalvaru v. State of Kerala",
            court="Supreme Court of India",
            year=1973,
            text=(
                "The basic structure doctrine is the bedrock of constitutional jurisprudence in India. "
                "Parliament's amending power under Article 368 of the Constitution of India is not unlimited and "
                "cannot be exercised to emasculate or alter the basic structure or essential features of the Constitution. "
                "Judicial review is a basic feature. Fundamental rights under Article 14, Article 19, and Article 21 "
                "form part of the constitutional scheme. (1973) 4 SCC 225, AIR 1973 SC 1461."
            ),
            cited_cases=[],
            cited_statutes=["CONST:ART_368", "CONST:ART_14", "CONST:ART_19", "CONST:ART_21"],
            ratio_summary="Parliament cannot alter the basic structure of the Constitution under Article 368."
        ),
        LegalCase(
            doc_id="SC_1978_MANEKA",
            title="Maneka Gandhi v. Union of India",
            court="Supreme Court of India",
            year=1978,
            text=(
                "Procedure established by law under Article 21 of the Constitution of India must be just, fair, and reasonable, "
                "not arbitrary, fanciful or oppressive. The right to travel abroad is part of personal liberty under Article 21. "
                "Articles 14, 19, and 21 form a golden triangle and are not mutually exclusive water-tight compartments. "
                "Natural justice and audi alteram partem must be read into administrative actions affecting liberty. "
                "Citing Kesavananda Bharati (1973) 4 SCC 225. AIR 1978 SC 597, (1978) 1 SCC 248."
            ),
            cited_cases=["SC_1973_KESHAVANANDA"],
            cited_statutes=["CONST:ART_21", "CONST:ART_14", "CONST:ART_19"],
            ratio_summary="Article 21 requires substantive due process: procedure must be just, fair, and reasonable."
        ),
        LegalCase(
            doc_id="SC_1980_BACHAN_SINGH",
            title="Bachan Singh v. State of Punjab",
            court="Supreme Court of India",
            year=1980,
            text=(
                "The constitutional validity of the death penalty for murder under Section 302 of the Indian Penal Code "
                "read with Section 354(3) of the Code of Criminal Procedure is upheld. "
                "The extreme penalty of death can be inflicted only in the 'rarest of rare cases' when the alternative option "
                "of life imprisonment is unquestionably foreclosed. The court must balance aggravating and mitigating circumstances "
                "of both the crime and the criminal. (1980) 2 SCC 684, AIR 1980 SC 898."
            ),
            cited_cases=["SC_1978_MANEKA"],
            cited_statutes=["STATUTE:IPC:SEC_302", "STATUTE:CrPC:SEC_354", "CONST:ART_21"],
            ratio_summary="Capital punishment limited to rarest of rare cases with mitigating-aggravating balance."
        ),
        LegalCase(
            doc_id="SC_1983_MACHHI_SINGH",
            title="Machhi Singh and Others v. State of Punjab",
            court="Supreme Court of India",
            year=1983,
            text=(
                "Applying the rarest of rare doctrine enunciated in Bachan Singh v. State of Punjab (1980) 2 SCC 684. "
                "Guidelines laid down for awarding death penalty under Section 302 IPC. Considerations include manner of commission "
                "of murder, motive, anti-social nature of crime, magnitude of crime, and personality of victim. "
                "Life imprisonment is the rule and capital sentence is an exception. (1983) 3 SCC 470, AIR 1983 SC 957."
            ),
            cited_cases=["SC_1980_BACHAN_SINGH"],
            cited_statutes=["STATUTE:IPC:SEC_302", "CONST:ART_21"],
            ratio_summary="Formulated five-point guidelines expanding the Bachan Singh rarest of rare sentencing framework."
        ),
        LegalCase(
            doc_id="SC_1987_MC_MEHTA_OLEUM",
            title="M.C. Mehta and Another v. Union of India (Oleum Gas Leak Case)",
            court="Supreme Court of India",
            year=1987,
            text=(
                "Principle of absolute liability laid down for hazardous and inherently dangerous industrial activities. "
                "The enterprise owes an absolute and non-delegable duty to the community to ensure no harm results. "
                "The rule in Rylands v. Fletcher with its exceptions does not apply in India. Article 32 of the Constitution "
                "empowers the court to award compensation for violation of fundamental rights under Article 21. "
                "(1987) 1 SCC 395, AIR 1987 SC 1086."
            ),
            cited_cases=[],
            cited_statutes=["CONST:ART_21", "CONST:ART_32"],
            ratio_summary="Absolute liability for hazardous industries without Rylands v Fletcher exceptions."
        ),
        LegalCase(
            doc_id="SC_1996_VELLORE_CITIZENS",
            title="Vellore Citizens Welfare Forum v. Union of India",
            court="Supreme Court of India",
            year=1996,
            text=(
                "Sustainable development, Precautionary Principle, and Polluter Pays Principle are essential features "
                "of Indian environmental law derived from Article 21 of the Constitution of India. "
                "Remediation of damaged environment and compensation to victims are part of the liability of polluting tanneries. "
                "Citing M.C. Mehta Oleum Gas case (1987) 1 SCC 395. (1996) 5 SCC 647, AIR 1996 SC 2715."
            ),
            cited_cases=["SC_1987_MC_MEHTA_OLEUM"],
            cited_statutes=["CONST:ART_21", "CONST:ART_32"],
            ratio_summary="Precautionary principle and polluter pays principle integrated into Article 21."
        ),
        LegalCase(
            doc_id="SC_2017_PUTTASWAMY",
            title="Justice K.S. Puttaswamy (Retd.) and Another v. Union of India",
            court="Supreme Court of India",
            year=2017,
            text=(
                "The right to privacy is a fundamental right protected under Article 21 and the entirety of Part III "
                "of the Constitution of India. Informational privacy, bodily autonomy, and spatial privacy are core aspects. "
                "Any invasion of privacy must satisfy the three-fold test of legality, legitimate state aim, and proportionality. "
                "Overruling MP Sharma and Kharak Singh to the extent they held privacy is not a fundamental right. "
                "Reaffirming Maneka Gandhi (1978) 1 SCC 248 and Kesavananda Bharati (1973) 4 SCC 225. (2017) 10 SCC 1."
            ),
            cited_cases=["SC_1973_KESHAVANANDA", "SC_1978_MANEKA"],
            cited_statutes=["CONST:ART_21", "CONST:ART_14", "CONST:ART_19"],
            ratio_summary="Right to privacy is an intrinsic part of the right to life and liberty under Article 21."
        ),
        LegalCase(
            doc_id="SC_2018_NAVTEJ_JOHAR",
            title="Navtej Singh Johar and Others v. Union of India",
            court="Supreme Court of India",
            year=2018,
            text=(
                "Section 377 of the Indian Penal Code, insofar as it criminalizes consensual sexual acts between adults in private, "
                "is unconstitutional as it violates Article 14, Article 15, Article 19, and Article 21 of the Constitution of India. "
                "Constitutional morality trumps majoritarian popular morality. The right to personal autonomy, dignity, and choice "
                "of identity are protected under the privacy doctrine. Citing Puttaswamy (2017) 10 SCC 1 and Maneka Gandhi. "
                "(2018) 10 SCC 1, AIR 2018 SC 4321."
            ),
            cited_cases=["SC_2017_PUTTASWAMY", "SC_1978_MANEKA"],
            cited_statutes=["STATUTE:IPC:SEC_377", "CONST:ART_14", "CONST:ART_15", "CONST:ART_19", "CONST:ART_21"],
            ratio_summary="Consensual adult sexual conduct decriminalized; Section 377 IPC declared unconstitutional in part."
        ),
        LegalCase(
            doc_id="SC_2015_SHREYA_SINGHAL",
            title="Shreya Singhal v. Union of India",
            court="Supreme Court of India",
            year=2015,
            text=(
                "Section 66A of the Information Technology Act, 2000 is struck down in its entirety as being violative of "
                "the freedom of speech and expression guaranteed under Article 19(1)(a) of the Constitution of India. "
                "The provision suffers from the vice of vagueness and overbreadth, creating a chilling effect on speech. "
                "Discussion, advocacy, and incitement distinguished. Clear and present danger standard examined. "
                "(2015) 5 SCC 1, AIR 2015 SC 1523."
            ),
            cited_cases=["SC_1978_MANEKA"],
            cited_statutes=["CONST:ART_19", "CONST:ART_21"],
            ratio_summary="Section 66A IT Act struck down for violating freedom of speech under Article 19(1)(a)."
        ),
        LegalCase(
            doc_id="SC_2019_ANURADHA_BHASIN",
            title="Anuradha Bhasin v. Union of India",
            court="Supreme Court of India",
            year=2020,
            text=(
                "Freedom of speech and expression and freedom to practice any profession over the medium of the internet "
                "is constitutionally protected under Article 19(1)(a) and Article 19(1)(g) of the Constitution of India. "
                "Any restriction or suspension of internet services under Section 144 of the Code of Criminal Procedure "
                "must pass the test of necessity and proportionality. Indefinite suspension is impermissible. "
                "Citing Shreya Singhal (2015) 5 SCC 1 and Puttaswamy (2017) 10 SCC 1. (2020) 3 SCC 637."
            ),
            cited_cases=["SC_2015_SHREYA_SINGHAL", "SC_2017_PUTTASWAMY"],
            cited_statutes=["CONST:ART_19", "STATUTE:CrPC:SEC_144"],
            ratio_summary="Internet access is protected under Article 19; suspension orders must be proportional and published."
        )
    ]

    queries = [
        LegalQuery(
            query_id="QRY_01_DEATH_PENALTY",
            facts=(
                "The accused was convicted under Section 302 IPC for brutal multiple homicides in a rural dispute. "
                "The trial court and High Court awarded the death sentence. On appeal to the Supreme Court, counsel argues "
                "that the sentencing judge did not properly record special reasons or weigh mitigating circumstances such as age "
                "and possibility of reformation under Section 354(3) CrPC before imposing capital punishment."
            ),
            issues="Whether the imposition of death sentence satisfies the rarest of rare doctrine and mitigating factor balancing test.",
            cited_statutes=["STATUTE:IPC:SEC_302", "STATUTE:CrPC:SEC_354", "CONST:ART_21"]
        ),
        LegalQuery(
            query_id="QRY_02_PRIVACY_SURVEILLANCE",
            facts=(
                "A state government enacted facial recognition surveillance at public transit hubs without statutory backing. "
                "A citizen petitioner challenges the surveillance program under Article 32, claiming it infringes informational "
                "privacy, bodily autonomy, and freedom of movement without satisfying the three-fold proportionality test."
            ),
            issues="Whether mass facial recognition surveillance without legislation violates the fundamental right to privacy under Article 21.",
            cited_statutes=["CONST:ART_21", "CONST:ART_19", "CONST:ART_14"]
        ),
        LegalQuery(
            query_id="QRY_03_INDUSTRIAL_POLLUTION",
            facts=(
                "A chemical fertilizer manufacturing plant leaked toxic gas into neighboring residential colonies, causing "
                "respiratory illnesses and environmental contamination. The factory owners claim force majeure and sabotage by third parties, "
                "disclaiming strict liability. Victims file a writ petition demanding restitution and compensation under Article 21."
            ),
            issues="Whether an enterprise carrying on hazardous activities is subject to absolute liability without common law exceptions.",
            cited_statutes=["CONST:ART_21", "CONST:ART_32"]
        ),
        LegalQuery(
            query_id="QRY_04_ONLINE_SPEECH_BAN",
            facts=(
                "The district administration issued an indefinite blanket ban on broadband and mobile internet services "
                "under Section 144 of the CrPC to maintain public order. Journalists and online merchants argue this arbitrary "
                "restriction infringes their freedom of speech, expression, and trade under Article 19."
            ),
            issues="Whether indefinite suspension of internet services violates freedom of speech under Article 19(1)(a) and the proportionality doctrine.",
            cited_statutes=["CONST:ART_19", "STATUTE:CrPC:SEC_144"]
        )
    ]

    qrels = [
        # Query 1: Death penalty -> Bachan Singh (primary), Machhi Singh (primary), Maneka Gandhi (secondary context)
        Qrel(query_id="QRY_01_DEATH_PENALTY", doc_id="SC_1980_BACHAN_SINGH", relevance=2),
        Qrel(query_id="QRY_01_DEATH_PENALTY", doc_id="SC_1983_MACHHI_SINGH", relevance=2),
        Qrel(query_id="QRY_01_DEATH_PENALTY", doc_id="SC_1978_MANEKA", relevance=1),

        # Query 2: Privacy / Surveillance -> Puttaswamy (primary), Maneka Gandhi (primary), Navtej Johar (secondary)
        Qrel(query_id="QRY_02_PRIVACY_SURVEILLANCE", doc_id="SC_2017_PUTTASWAMY", relevance=2),
        Qrel(query_id="QRY_02_PRIVACY_SURVEILLANCE", doc_id="SC_1978_MANEKA", relevance=2),
        Qrel(query_id="QRY_02_PRIVACY_SURVEILLANCE", doc_id="SC_2018_NAVTEJ_JOHAR", relevance=1),
        Qrel(query_id="QRY_02_PRIVACY_SURVEILLANCE", doc_id="SC_1973_KESHAVANANDA", relevance=1),

        # Query 3: Environmental pollution -> M.C. Mehta Oleum (primary), Vellore Citizens (primary)
        Qrel(query_id="QRY_03_INDUSTRIAL_POLLUTION", doc_id="SC_1987_MC_MEHTA_OLEUM", relevance=2),
        Qrel(query_id="QRY_03_INDUSTRIAL_POLLUTION", doc_id="SC_1996_VELLORE_CITIZENS", relevance=2),

        # Query 4: Online Speech Ban -> Shreya Singhal (primary), Anuradha Bhasin (primary), Puttaswamy (secondary)
        Qrel(query_id="QRY_04_ONLINE_SPEECH_BAN", doc_id="SC_2019_ANURADHA_BHASIN", relevance=2),
        Qrel(query_id="QRY_04_ONLINE_SPEECH_BAN", doc_id="SC_2015_SHREYA_SINGHAL", relevance=2),
        Qrel(query_id="QRY_04_ONLINE_SPEECH_BAN", doc_id="SC_2017_PUTTASWAMY", relevance=1)
    ]

    return cases, queries, qrels


def save_dataset_to_disk(
    cases: List[LegalCase],
    queries: List[LegalQuery],
    qrels: List[Qrel],
    output_dir: str
) -> None:
    """Saves corpus, queries, and qrels into standardized disk formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Save corpus to corpus.jsonl
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")

    # Save queries to queries.jsonl
    queries_path = os.path.join(output_dir, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query.to_dict()) + "\n")

    # Save qrels to qrels.tsv (TREC format)
    qrels_path = os.path.join(output_dir, "qrels.tsv")
    with open(qrels_path, "w", encoding="utf-8") as f:
        for qrel in qrels:
            f.write(qrel.to_trec_line())

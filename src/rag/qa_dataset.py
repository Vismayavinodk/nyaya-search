"""
Self-written curated benchmark test set (35 questions) for Legal Question Answering.
Includes:
- 25 Answerable Precedent Questions with ground-truth facts, citations, and statutory references.
- 10 Unanswerable / Weak-Evidence Adversarial Questions to evaluate hallucination prevention & refusal.
"""

from __future__ import annotations
from typing import List, Dict
import json
import os

from src.rag.schema import LegalQAExample


BENCHMARK_EXAMPLES: List[Dict] = [
    # =========================================================================
    # Category 1: Constitutional Landmark Precedents (Answerable)
    # =========================================================================
    {
        "question_id": "QA_001",
        "question": "Does Parliament have unlimited power to amend any part of the Constitution under Article 368?",
        "gold_answer": "No, Parliament's amending power under Article 368 is limited and cannot damage or destroy the basic structure of the Constitution.",
        "gold_doc_ids": ["case_kesavananda", "case_minerva_mills"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 368"],
        "evidence_text": "In Kesavananda Bharati v. State of Kerala (1973), the Supreme Court ruled that Article 368 does not enable Parliament to alter the basic structure or framework of the Constitution."
    },
    {
        "question_id": "QA_002",
        "question": "Is the Right to Privacy recognized as a fundamental right under the Indian Constitution, and under which Article?",
        "gold_answer": "Yes, the Right to Privacy is a fundamental right protected under Article 21 and Part III of the Constitution.",
        "gold_doc_ids": ["case_puttaswamy"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 21"],
        "evidence_text": "In K.S. Puttaswamy v. Union of India (2017), a 9-judge bench unanimously held that privacy is an intrinsic part of the right to life and personal liberty under Article 21."
    },
    {
        "question_id": "QA_003",
        "question": "What standard must a law depriving personal liberty meet under Article 21 following Maneka Gandhi?",
        "gold_answer": "The law must prescribe a procedure that is just, fair, and reasonable, and not arbitrary, fanciful, or oppressive.",
        "gold_doc_ids": ["case_maneka_gandhi"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 21", "Article 14", "Article 19"],
        "evidence_text": "In Maneka Gandhi v. Union of India (1978), the Supreme Court held that procedure established by law under Article 21 must be just, fair and reasonable, establishing the golden triangle with Articles 14 and 19."
    },
    {
        "question_id": "QA_004",
        "question": "Are constitutional amendments inserted into the Ninth Schedule immune from judicial review after April 24, 1973?",
        "gold_answer": "No, laws added to the Ninth Schedule post April 24, 1973 are open to judicial review if they violate fundamental rights forming part of the basic structure.",
        "gold_doc_ids": ["case_ir_coelho"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 31B", "Ninth Schedule"],
        "evidence_text": "In I.R. Coelho v. State of Tamil Nadu (2007), a 9-judge bench held that all statutory amendments included in the Ninth Schedule after 24 April 1973 must pass the basic structure test."
    },
    {
        "question_id": "QA_005",
        "question": "Is secularism considered a basic feature of the Indian Constitution according to the Supreme Court?",
        "gold_answer": "Yes, secularism is an essential part of the basic structure of the Constitution, and state governments acting against secularism are liable to action under Article 356.",
        "gold_doc_ids": ["case_sr_bommai"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 356", "Preamble"],
        "evidence_text": "In S.R. Bommai v. Union of India (1994), the Supreme Court confirmed that secularism is a basic feature of the Constitution and state actions undermining it justify President's Rule under Article 356."
    },
    {
        "question_id": "QA_006",
        "question": "What is the ceiling on reservation percentages laid down in the Mandal case?",
        "gold_answer": "Total reservations under Article 16(4) should ordinarily not exceed 50%, and the creamy layer must be excluded.",
        "gold_doc_ids": ["case_indira_sawhney"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 16(4)"],
        "evidence_text": "In Indra Sawhney v. Union of India (1992), the 9-judge bench upheld 27% OBC reservation subject to exclusion of the creamy layer and a general cap of 50% on reservations."
    },
    {
        "question_id": "QA_007",
        "question": "Does Article 21 encompass the right to livelihood according to Supreme Court jurisprudence?",
        "gold_answer": "Yes, the right to livelihood is an integral facet of the right to life guaranteed under Article 21.",
        "gold_doc_ids": ["case_olga_tellis"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 21"],
        "evidence_text": "In Olga Tellis v. Bombay Municipal Corporation (1985), the Constitution Bench held that the right to livelihood is an essential facet of the right to life under Article 21."
    },
    {
        "question_id": "QA_008",
        "question": "Can reasons for an administrative order be supplemented subsequently through explanatory counter-affidavits?",
        "gold_answer": "No, an administrative order must be judged strictly by the reasons recorded therein at the time of issuance and cannot be validated by subsequent affidavits.",
        "gold_doc_ids": ["case_mohinder_gill"],
        "is_answerable": True,
        "domain": "Administrative Law",
        "cited_statutes": ["Article 324"],
        "evidence_text": "In Mohinder Singh Gill v. Chief Election Commissioner (1977), the Supreme Court ruled that public orders cannot be construed in the light of subsequent explanations given by the officer."
    },
    {
        "question_id": "QA_009",
        "question": "Is the practice of instant Triple Talaq (Talaq-e-Biddat) legally valid in India?",
        "gold_answer": "No, instant Triple Talaq was declared unconstitutional, void, and manifestly arbitrary under Article 14.",
        "gold_doc_ids": ["case_shayara_bano"],
        "is_answerable": True,
        "domain": "Family & Constitutional Law",
        "cited_statutes": ["Article 14"],
        "evidence_text": "In Shayara Bano v. Union of India (2017), the Supreme Court set aside the practice of Talaq-e-Biddat as unconstitutional and violative of Article 14."
    },
    {
        "question_id": "QA_010",
        "question": "Is consensual same-sex sexual activity between adults criminalized under Section 377 of the Indian Penal Code?",
        "gold_answer": "No, Section 377 was read down to decriminalize consensual adult same-sex relations.",
        "gold_doc_ids": ["case_navtej_johar"],
        "is_answerable": True,
        "domain": "Criminal & Constitutional Law",
        "cited_statutes": ["Section 377 IPC", "Article 14", "Article 21"],
        "evidence_text": "In Navtej Singh Johar v. Union of India (2018), the Supreme Court unanimously decriminalized consensual sexual intercourse between adults of the same sex under Section 377 IPC."
    },

    # =========================================================================
    # Category 2: Criminal Law & Procedure Safeguards (Answerable)
    # =========================================================================
    {
        "question_id": "QA_011",
        "question": "Is registration of an FIR mandatory upon receiving information disclosing a cognizable offense?",
        "gold_answer": "Yes, registration of an FIR under Section 154 CrPC is mandatory if the information discloses commission of a cognizable offence.",
        "gold_doc_ids": ["case_lalita_kumari"],
        "is_answerable": True,
        "domain": "Criminal Procedure",
        "cited_statutes": ["Section 154 CrPC"],
        "evidence_text": "In Lalita Kumari v. Govt. of U.P. (2013), the Constitution Bench held that Section 154 CrPC makes registration of FIR mandatory where cognizable offence is disclosed."
    },
    {
        "question_id": "QA_012",
        "question": "What safeguards must police officers follow during arrest to prevent custodial violence and torture?",
        "gold_answer": "Police must wear visible identification, prepare an arrest memo, inform a relative or friend, and conduct medical examinations every 48 hours.",
        "gold_doc_ids": ["case_dk_basu"],
        "is_answerable": True,
        "domain": "Criminal Procedure",
        "cited_statutes": ["Article 21", "Article 22"],
        "evidence_text": "In D.K. Basu v. State of West Bengal (1997), the Supreme Court prescribed 11 mandatory guidelines to be followed by law enforcement during arrests and detention."
    },
    {
        "question_id": "QA_013",
        "question": "Are automatic arrests permitted in offenses punishable with imprisonment up to 7 years such as Section 498A IPC?",
        "gold_answer": "No, police must not automatically arrest without satisfying the necessity conditions under Section 41 CrPC and issuing a notice under Section 41A.",
        "gold_doc_ids": ["case_arnesh_kumar"],
        "is_answerable": True,
        "domain": "Criminal Procedure",
        "cited_statutes": ["Section 41 CrPC", "Section 41A CrPC", "Section 498A IPC"],
        "evidence_text": "In Arnesh Kumar v. State of Bihar (2014), the Supreme Court held that arrests under Section 498A IPC and offences punishable up to 7 years must satisfy conditions of Section 41 CrPC."
    },
    {
        "question_id": "QA_014",
        "question": "Does an undertrial prisoner have a fundamental right to a speedy trial under the Indian Constitution?",
        "gold_answer": "Yes, the right to a speedy trial is a fundamental right implicit in Article 21.",
        "gold_doc_ids": ["case_hussainara_khatoon"],
        "is_answerable": True,
        "domain": "Criminal Procedure",
        "cited_statutes": ["Article 21"],
        "evidence_text": "In Hussainara Khatoon v. Home Secretary, State of Bihar (1979), the Supreme Court recognized that right to a speedy public trial is an inalienable component of Article 21."
    },
    {
        "question_id": "QA_015",
        "question": "Is solitary confinement or putting prisoners in bar fetters permissible without judicial authorization?",
        "gold_answer": "No, putting prisoners in bar fetters or punitive solitary confinement without judicial sanction violates human dignity under Articles 14, 19, and 21.",
        "gold_doc_ids": ["case_sunil_batra"],
        "is_answerable": True,
        "domain": "Prisoners Rights",
        "cited_statutes": ["Article 21", "Prisons Act"],
        "evidence_text": "In Sunil Batra v. Delhi Administration (1980), the Supreme Court prohibited inhuman treatment, bar fetters, and arbitrary solitary confinement of prisoners."
    },
    {
        "question_id": "QA_016",
        "question": "Is the State liable to pay monetary compensation under public law for custodial deaths causing violation of Article 21?",
        "gold_answer": "Yes, the Supreme Court and High Courts can award public law monetary compensation for custodial deaths resulting from human rights violations.",
        "gold_doc_ids": ["case_nilabati_behera"],
        "is_answerable": True,
        "domain": "Constitutional Torts",
        "cited_statutes": ["Article 32", "Article 226", "Article 21"],
        "evidence_text": "In Nilabati Behera v. State of Orissa (1993), the Supreme Court affirmed that sovereign immunity is inapplicable to public law remedies awarding compensation for custodial deaths."
    },
    {
        "question_id": "QA_017",
        "question": "Is the offense of adultery under Section 497 of the Indian Penal Code constitutionally valid?",
        "gold_answer": "No, Section 497 IPC was struck down as unconstitutional for treating women as chattel and violating Articles 14 and 21.",
        "gold_doc_ids": ["case_joseph_shine"],
        "is_answerable": True,
        "domain": "Criminal Law",
        "cited_statutes": ["Section 497 IPC", "Article 14", "Article 21"],
        "evidence_text": "In Joseph Shine v. Union of India (2018), the Supreme Court struck down Section 497 IPC as unconstitutional, arbitrary, and discriminatory against women."
    },

    # =========================================================================
    # Category 3: Environmental, Labour, Torts & Special Laws (Answerable)
    # =========================================================================
    {
        "question_id": "QA_018",
        "question": "What is the standard of liability for hazardous and inherently dangerous industrial enterprises in India?",
        "gold_answer": "An enterprise engaged in hazardous activities owes an absolute and non-delegable duty to ensure no harm occurs, not subject to the exceptions in Rylands v. Fletcher.",
        "gold_doc_ids": ["case_mc_mehta"],
        "is_answerable": True,
        "domain": "Environmental Law & Torts",
        "cited_statutes": ["Article 21", "Article 32"],
        "evidence_text": "In M.C. Mehta v. Union of India (Oleum Gas Leak case, 1987), the Supreme Court developed the doctrine of Absolute Liability, holding that enterprise liability is unlimited and strict."
    },
    {
        "question_id": "QA_019",
        "question": "What guidelines govern the prevention of sexual harassment of women at the workplace in the absence of legislation?",
        "gold_answer": "The Vishaka guidelines mandate Internal Complaints Committees, preventive measures, and disciplinary action against sexual harassment at workplaces.",
        "gold_doc_ids": ["case_vishaka"],
        "is_answerable": True,
        "domain": "Labour & Gender Law",
        "cited_statutes": ["Article 14", "Article 19", "Article 21", "CEDAW"],
        "evidence_text": "In Vishaka v. State of Rajasthan (1997), the Supreme Court laid down legally binding norms and guidelines prohibiting sexual harassment at the workplace."
    },
    {
        "question_id": "QA_020",
        "question": "Is passive euthanasia and the right to execute an advance medical directive (living will) permissible in India?",
        "gold_answer": "Yes, passive euthanasia is legally recognized under Article 21, allowing competent adults to make advance medical directives to withdraw life support.",
        "gold_doc_ids": ["case_common_cause"],
        "is_answerable": True,
        "domain": "Medical Law & Article 21",
        "cited_statutes": ["Article 21"],
        "evidence_text": "In Common Cause v. Union of India (2018), the Constitution Bench held that the right to die with dignity is an inseparable facet of Article 21 and recognized living wills."
    },
    {
        "question_id": "QA_021",
        "question": "What doctrine determines whether an enactment relates to a state list entry or a union list entry when legislative competence is challenged?",
        "gold_answer": "The Doctrine of Pith and Substance examines the true nature and character of the legislation to ascertain its primary subject matter.",
        "gold_doc_ids": ["case_balsara"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 246", "Seventh Schedule"],
        "evidence_text": "In State of Bombay v. F.N. Balsara (1951), the Supreme Court applied the doctrine of pith and substance to determine legislative competence under the Seventh Schedule."
    },
    {
        "question_id": "QA_022",
        "question": "How does the Supreme Court define bonded labour under Article 23 of the Constitution?",
        "gold_answer": "Any labour performed where wages paid are less than the statutory minimum wage constitutes forced labour under Article 23.",
        "gold_doc_ids": ["case_bandhua_mukti"],
        "is_answerable": True,
        "domain": "Labour Law",
        "cited_statutes": ["Article 23", "Bonded Labour System Act"],
        "evidence_text": "In Bandhua Mukti Morcha v. Union of India (1984), the Supreme Court held that failure to provide minimum wage amounts to forced labour prohibited under Article 23."
    },
    {
        "question_id": "QA_023",
        "question": "Are private unaided minority educational institutions required to provide 25% quota under the Right to Education Act?",
        "gold_answer": "No, unaided minority educational institutions are exempt from the 25% admission quota under the Right to Education Act.",
        "gold_doc_ids": ["case_pramati"],
        "is_answerable": True,
        "domain": "Education Law",
        "cited_statutes": ["Article 21A", "Article 30(1)", "RTE Act"],
        "evidence_text": "In Pramati Educational Trust v. Union of India (2014), the 5-judge bench held that the RTE Act does not apply to private unaided minority institutions under Article 30(1)."
    },
    {
        "question_id": "QA_024",
        "question": "What was the initial holding in A.K. Gopalan regarding the relationship between Article 19 and Article 21, and was it later overruled?",
        "gold_answer": "A.K. Gopalan held that Articles 19 and 21 are mutually exclusive compartments, which was later overruled by Maneka Gandhi.",
        "gold_doc_ids": ["case_ak_gopalan", "case_maneka_gandhi"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 19", "Article 21"],
        "evidence_text": "In A.K. Gopalan v. State of Madras (1950), the court took a narrow silo view of fundamental rights, which was decisively overruled in Maneka Gandhi (1978)."
    },
    {
        "question_id": "QA_025",
        "question": "Did Golak Nath hold that constitutional amendments cannot abridge Fundamental Rights, and was this stance modified?",
        "gold_answer": "Yes, Golak Nath held Parliament could not abridge Fundamental Rights, but this was modified in Kesavananda Bharati which upheld amendment power subject to the basic structure doctrine.",
        "gold_doc_ids": ["case_golak_nath", "case_kesavananda"],
        "is_answerable": True,
        "domain": "Constitutional Law",
        "cited_statutes": ["Article 13(2)", "Article 368"],
        "evidence_text": "In I.C. Golak Nath v. State of Punjab (1967), an 11-judge bench held Fundamental Rights were unamendable, which was superseded by the 13-judge bench in Kesavananda (1973)."
    },

    # =========================================================================
    # Category 4: Unanswerable / Weak-Evidence / Refusal Questions (10 Examples)
    # The system MUST refuse when evidence is weak or irrelevant.
    # =========================================================================
    {
        "question_id": "QA_REFUSAL_001",
        "question": "What is the penalty for exceeding the atmospheric re-entry speed limit on Mars according to Supreme Court precedent?",
        "gold_answer": "REFUSAL: Insufficient evidence in legal precedent corpus.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_002",
        "question": "What did the Supreme Court hold regarding copyright protection for generative artificial intelligence music in the 1973 Kesavananda judgment?",
        "gold_answer": "REFUSAL: The 1973 Kesavananda judgment did not address artificial intelligence or generative music.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_003",
        "question": "Under what statutory exceptions can a High Court judge lawfully accept luxury personal gifts from active trial litigants?",
        "gold_answer": "REFUSAL: There is no legal precedent or statute permitting judicial officers to accept gifts from litigants.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_004",
        "question": "What is the limitation period under the 1872 Limitation Act for asserting ownership over Bitcoin and Ethereum blockchain tokens?",
        "gold_answer": "REFUSAL: Blockchain cryptocurrency tokens are not contemplated in 19th century precedent.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_005",
        "question": "How many years of mandatory rigorous imprisonment are prescribed in the Constitution for an eligible citizen failing to cast a vote in general elections?",
        "gold_answer": "REFUSAL: Voting is not legally compulsory in India and the Constitution prescribes no penal imprisonment for non-voting.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_006",
        "question": "What customs tariff schedule did the Minerva Mills decision establish for extraterrestrial mining imports?",
        "gold_answer": "REFUSAL: Minerva Mills concerned constitutional amendment powers under Article 368, not extraterrestrial commerce.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_007",
        "question": "Can an executive sub-divisional magistrate grant bail for deep-sea piracy under Article 19(1)(a) of the Constitution?",
        "gold_answer": "REFUSAL: Insufficient legal evidence; Article 19(1)(a) protects freedom of speech and does not govern maritime piracy bail powers.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_008",
        "question": "Which section of the Indian Penal Code regulates the maximum allowable decibel level for commercial supersonic passenger aircraft?",
        "gold_answer": "REFUSAL: Supersonic aircraft noise is not regulated under the Indian Penal Code.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_009",
        "question": "Does Article 14 guarantee private citizens a fundamental right to own and operate nuclear attack submarines?",
        "gold_answer": "REFUSAL: There is no fundamental right under Article 14 or any precedent permitting private nuclear armament.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    },
    {
        "question_id": "QA_REFUSAL_010",
        "question": "Under what circumstances does the Indian Evidence Act require judicial notice of astrological horoscopes in murder trials?",
        "gold_answer": "REFUSAL: The Indian Evidence Act does not recognize astrological horoscopes as admissible judicial notice facts in criminal trials.",
        "gold_doc_ids": [],
        "is_answerable": False,
        "domain": "Unsupported/Adversarial",
        "cited_statutes": [],
        "evidence_text": None
    }
]


class RAGQADataset:
    """
    Manages loading, serialization, and sampling of the legal QA benchmark.
    """

    @staticmethod
    def load_benchmark() -> List[LegalQAExample]:
        """Loads the self-written 35-question legal QA benchmark."""
        return [LegalQAExample(**item) for item in BENCHMARK_EXAMPLES]

    @staticmethod
    def save_to_json(filepath: str) -> None:
        """Exports benchmark to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(BENCHMARK_EXAMPLES, f, indent=2, ensure_ascii=False)

**AI Solution Overview**



**The theme of this project is an AI solution for AI across industries, and MMG AI targets the finance industry. Rather than build a black box predictive model, MMG AI delivers a transparent, rule based early warning system: every flag a customer receives can be traced back to specific, bank-configurable thresholds on their account data (credit-card utilisation, late payments, balance levels, balance decline, overdraft, and loan exposure). That transparency matters in a regulated financial setting, where an analyst and a regulator must be able to explain exactly why a customer was flagged.**



**The delivered system has three parts:**



**- A scoring engine (bank\_distress\_bot.py) that assesses every customer against seven weighted rules and assigns a status of NO FLAG, EARLY WARNING or FINANCIAL DISTRESS, with every point fully explainable.**

**- A chat assistant, available in the terminal or a browser (sample.html, via --serve), that lets an employee query the engine in plain English ("Check customer 10452", "Find customers at risk", "How does flagging work?"), record flag reviews and approvals, and see a full audit trail. An optional Google Gemini model can rephrase a question it doesn't recognise into a supported one; it never sees customer data and never decides or overrides a flag.**

**- A reporting tool (charts.py) with seven chart types covering class distributions, demographic fairness, customer feedback, trends over time, and honestly labelled machine learning diagnostics (ROC curves, feature importance, a confusion matrix) that test what a Random Forest or Logistic Regression model trained on the same account fields can, and cannot, tell you beyond the rules themselves.**





**Business Understanding**

&#x20;

**Business Background**



**Monetary Mindset Group (MMG) is a financial services technology company that partners with retail banks to improve**

**customer financial wellness. The company holds a comprehensive banking database of 5,000 customers covering**

**demographics, accounts, transactions, loans, credit cards and customer feedback (2023 transaction year). Management**

**identified that distress events (negative balances, maxed-out credit) and unusual transactions are currently identified**

**manually and too late. MMG AI is the proposed rule-based, auditable early warning system.**



**Business Objectives**



**- Flag customers already showing multiple, weighted signs of financial distress, using transparent, bank configurable**

**rules rather than a black box score.**

**- Surface the bank's own existing anomaly marker alongside the rule-based flags so analysts review both together. The**

**system does not independently detect anomalies with an unsupervised model that is noted as future work in Section**

**12.**

**- Chart monthly trends in flagged customers, average risk score, balance and transaction volume to support capacity**

**and cash flow planning; a descriptive trend view, not a statistical forecast.**

**- Summarise customer feedback (type mix, resolution outcomes, time to resolve, and, where free text exists, the terms**

**that most distinguish flagged from unflagged customers) to help prioritise service recovery.**

**- Provide always on self service through a chatbot that answers account, flag explanation and support suggestion**

**questions, and records employee reviews of each flag.**





**Business Success Criteria**



**- Every flag is fully explainable: the chatbot's "why" and "rules" commands reproduce the exact points and thresholds**

**behind a customer's status, with no black-box step in the decision path.**

**- he bank's own anomaly marker (6.0% of the current 5,000-customer file) is surfaced for review rather than**

**recomputed, and its overlap with the rule-based flags is charted so analysts see where the two agree and disagree.**

**- Monthly trend charts (charts.py time\_series) track flagged customers, score, balance and transaction volume without**

**needing a forecasting model.**

**- he chatbot resolves at least ten common request types: check a customer, find at-risk customers, find customers with**

**multiple signals, filter by a specific signal, explain the rules, compare recent changes, suggest support, resolve a flag,**

**approve a resolution, and list more results without human intervention.**

**- The pipeline is Python only, open source, and fast: loading, scoring and auditing all 5,000 customers took a fraction of a**

**second in testing on a standard machine (well inside the 60-second target).**





**Requirements**

|**Type**|**Requirements**|
|-|-|
|Functional|Ingest the bank's customer or transaction CSV and compute every rule based indicator automatically. Produce a transparent, rule based risk score and status per customer. Let an employee query, filter and review flagged customers through a chat interface (terminal or browser). Provide chart based reporting: status or score distributions, demographic fairness, feedback analysis, time trends and ML diagnostics. Chatbot interface for employee queries (text; voice input in the browser via the Web Speech API).|
|Non-Functional|Performance: score a batch of 5,000 customers in well under a second on a standard laptop (measured). Maintainability: every threshold lives in one configuration dictionary; all seven chart types share one reporting script. Accuracy/Honesty: any machine learning diagnostic is evaluated on a held out test split, and its result is reported alongside a note on whether the target being predicted is independent of the rule engine's own inputs. Usability: plain-language command line and browser chat, plus saved PNG charts.|
|Technical|Python 3.10+, pandas, scikit-learn (diagnostic charts only), Matplotlib, NumPy, SQLite (built in, for the audit log and flag reviews), Python's built-in http.server (browser UI, no external web framework), optional Google Gemini API (question rephrasing only).|
|Data|Comprehensive\_Banking\_Database.csv the bank's own extract; no separate synthetic time-series was generated, the existing Transaction Date column is used directly for all trend charts.|





**Constraints**



* **Data Privacy: because scoring is rule-based rather than model trained, no customer data is used to fit a decision making model at all. The optional Random Forest or Logistic Regression diagnostics in charts.py are for internal explanation only and are never used to decide a flag.**
* **Infrastructure: the solution runs on standard hardware without any GPU dependency.**
* **Language: the prototype is English only, due to limited labelled multilingual data.**
* **Regulatory: must comply with South African POPIA and FICA regulations regarding financial advice**



&#x20;**Risks**

|Risks|Impact|Mitigation|
|-|-|-|
|Class imbalance in the ML diagnostics (only 101 of 5,000 customers, 2.0%, reach FINANCIAL DISTRESS)|A diagnostic model can look accurate while missing the minority class|class\_weight="balanced" in both diagnostic models; per-class recall is reported in the confusion-matrix chart, not just overall accuracy.|
|Feature/label leakage when a model is trained to reproduce the rule engine's own status labels|Inflated, meaningless accuracy (measured: 97.4% accuracy, ROC-AUC 0.998 for Random Forest on "status")|The code prints an explicit leakage note on every relevant chart and treats the result as "how the rules behave", never as a predictive result to act on|
|The same account features do not actually predict the bank's independent anomaly marker (measured: ROC-AUC ≈ 0.51, no better than chance)|A team could wrongly assume the rule-engine inputs also explain anomalies|Reported honestly in Section 9 rather than hidden; anomalies remain a separate, bank-supplied review queue.<br />|
|Scope creep (chatbot, ML diagnostics, feedback analysis, trend charts, browser UI)|Missed deadline|The rule engine and chat assistant were built and shipped first; charts and diagnostics were added as read-only, optional add-ons that never touch the CSV or the deployed decision path|







**Tools \& Tecniques**

|**Category**|**Tool/Technique**|**Purpose**|
|-|-|-|
|Language|Python 3.10+|Core development|
|Rule engine|Plain Python, pandas|Deterministic, auditable scoring no ML in the decision pathDeterministic, auditable scoring no ML in the decision path|
|ML library (diagnostics only)|scikit-learn (Random Forest, Logistic Regression)|Explaining/validating the rules on held-out data, never deciding a flag|
|NLP|scikit-learn CountVectorizer (bag of words)|Top term comparison between flagged and unflagged feedback|
|Visualisation|Matplotlib|All seven chart types: distributions, fairness, feedback, trends, ROC, importance, confusion matrix|
|Voice|Browser Web Speech API (sample.html)|Voice to text input only no server side speech library, no speech output|
|Persistence/audit|SQLite (Python's sqlite3, built in)|Data snapshots, flag reviews/approvals, full query audit log|
|Web server|Python's http.server (built in)|Serves the browser chat UI no external framework|
|LLM assist (optional)|Google Gemini API|Rephrases an unrecognised question into a supported one; never sees results or decides flags|
|Version control|Git, GitHub|Collaborative development and project tracking|





**Problem Definition** 



**What exactly is the problem?**



Banks in developing and underdeveloped regions lose significant revenue because customer financial distress is detected too late. The current process is entirely manual and reactive; a teller or analyst typically notices an issue only after money has already been lost. In MMG's banking dataset, a meaningful share of customers show high credit card utilisation, late payments, declining or overdrawn balances, and heavy loan exposure that manual monitoring misses in real time. The precise problem is the absence of a fast, transparent, automated system that screens every customer's account activity against a consistent set of rules and immediately explains why a customer was flagged.



**How relevant is it to the theme?**



This problem sits at the intersection of applied automation and local economic development. Reactive, purely manual review is no longer sufficient to protect modern financial ecosystems. By moving from manual oversight to consistent, always on, explainable rule based screeningbacked by a chat interface, analysts can actually query this problem directly. This problem directly addresses how practical automation can replace outdated manual tracking to solve structural revenue loss and systemic debt, without requiring the bank to trust an opaque model.



**How beneficial will it be in solving the problem?**



Solving this problem delivers a dual benefit to both the financial institution and the wider community:



* **For the Bank**: fewer undetected overdrafts and defaults translate directly into lower financial losses. Because every flag is explainable, staff can act on it immediately and defend the decision, improving operational stability and reducing risk exposure.
* **For Customers and the Community:** proactive early warnings, together with the chatbot's built in support suggestions per warning sign, allow account managers to step in with restructuring options, tailored payment plans, or financial education support before a household collapses into severe default. Preventing these financial crises improves local financial literacy, protects vulnerable households from predatory debt spirals, and builds trust in the local banking sector.


# Group Sequential Tests
**Terminate futile experiments early with planned interim analyses.**

This is a classic and powerful method where you pre-define several points for "interim analysis" during the experiment (e.g., every week, every 1000 users). At each point, you evaluate for statistical significance or futility (i.e., concluding that continuing the test is unlikely to yield a significant result).

* **Use Cases:**
    * When analysis timing is fixed, such as for weekly business reviews.
    * For structured experiments like clinical trials that require strict error control.
* **Key Features:**
    * **Spending Functions:** Supports various functions like O'Brien-Fleming (conservative) and Pocock (aggressive) to control the probability of early termination.
    * **Stopping for Efficacy or Futility:** Allows for early stopping based on conclusions of either effectiveness ("there is an effect") or futility ("unlikely to find a significant difference even if continued").

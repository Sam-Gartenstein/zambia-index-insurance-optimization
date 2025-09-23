# Zambia Weather Index Insurance Optimization

The goal of this project is to reduce basis risk in weather insurance contracts, building upon [earlier work](https://github.com/Sam-Gartenstein/zambia-drought-analysis) where we introduced the Payout Balance Index (PBI). In addition to K-means clustering, we calculate the PBI using Zambia’s two administrative levels along with camps developed by (name), and estimate the monetary value of insurance forgone due to mismatches between payouts and actual conditions. This is calculated by 

$$
\text{Forgone Payout} =
(\text{Payout Value}) \times (1 - \text{PBI}) \times (\text{Number of Pixels Per Zone})
$$  


After calculating the forgone payments, we calculate the marginal benefit per zone change (going from lower to higher zone numbers). That is, the marginal effect of moving from zone $i$ to zone $j$ ($i \to j$) is defined as:

$$
m_{i \to j} = \frac{\lvert FP_j - FP_i \rvert}{Z_j - Z_i},
$$

where $FP_k$ is the forgone payment at level $k$ and $Z_k$ is the number of zones at that level. The quantity $m_{i \to j}$ measures the absolute change in forgone payments per additional zone (ZMW per zone). Remove the absolute value if a signed measure is preferred.


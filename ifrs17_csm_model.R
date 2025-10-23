# ============================================================================
# Contractual Service Margin Calculation Model - Building Block Approach
# ============================================================================
# 
# Author: Judy Chang
# Objective: 本模型實現 IFRS 17 準則下的 CSM 計算，並同時支援 TW-ICS 資本適足性評估
# 
# 主要功能模組:
# 1. project_cashflows()         - 履行現金流量預測
# 2. calculate_discount_factors() - 折現因子計算
# 3. determine_risk_adjustment()  - 風險調整計算（資本成本法）
# 4. calculate_BBA_components()   - BBA 各組件計算與 CSM 推導
# 5. amortize_csm()               - CSM 攤銷計算
# 6. run_scenario()               - 敏感性分析框架
# ============================================================================

# ============================================================================
# 載入必要套件
# ============================================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(knitr)
  library(scales)
})

# ============================================================================
# 模組 1: 現金流量預測
# ============================================================================

#' 預測履行現金流量
#' 
#' 根據保單資料和精算假設，逐年預測未來現金流入與流出
#' 
#' @param policy_data 包含保單基本資料的列表
#' @param assumptions 包含精算與經濟假設的列表
#' @param proj_years 預測年期數
#' @return 包含逐年現金流量預測的資料框
#' @export
project_cashflows <- function(policy_data, assumptions, proj_years) {
  
  # 初始化結果資料框
  cf_projection <- tibble(
    year = 0:proj_years,
    policies_start_of_year = 0,
    premium_inflow = 0,
    death_outflow = 0,
    surrender_outflow = 0,
    expense_outflow = 0,
    net_cashflow = 0,
    policies_end_of_year = 0
  )
  
  # 設定初始狀態 (t=0)
  cf_projection$policies_end_of_year[1] <- policy_data$initial_policies
  
  # 逐年預測 (t=1 to proj_years)
  for (t in 1:proj_years) {
    current_row <- t + 1
    prev_row <- t
    
    # 取得當期精算假設
    mortality_rate <- assumptions$mortality_rate[t]
    surrender_rate <- assumptions$surrender_rate[t]
    
    # 計算期初有效保單數
    policies_boy <- cf_projection$policies_end_of_year[prev_row]
    cf_projection$policies_start_of_year[current_row] <- policies_boy
    
    # 如果無保單則跳出
    if (policies_boy == 0) break
    
    # 預測現金流入與流出
    # 保費收入
    prem_inflow <- policies_boy * policy_data$premium_per_policy
    cf_projection$premium_inflow[current_row] <- prem_inflow
    
    # 死亡理賠支出
    death_claims <- policies_boy * mortality_rate * policy_data$sum_assured_per_policy
    cf_projection$death_outflow[current_row] <- death_claims
    
    # 脫退給付支出
    surrender_payouts <- policies_boy * (1 - mortality_rate) * surrender_rate * 
                         policy_data$cash_value_per_policy[t]
    cf_projection$surrender_outflow[current_row] <- surrender_payouts
    
    # 費用支出
    acquisition_expense <- ifelse(t == 1, 
                                  policies_boy * assumptions$acq_expense_per_policy, 
                                  0)
    maintenance_expense <- policies_boy * assumptions$maint_expense_per_policy
    commission_expense <- prem_inflow * assumptions$commission_rate[t]
    total_expense <- acquisition_expense + maintenance_expense + commission_expense
    cf_projection$expense_outflow[current_row] <- total_expense
    
    # 計算淨現金流量
    cf_projection$net_cashflow[current_row] <- prem_inflow - death_claims - 
                                               surrender_payouts - total_expense
    
    # 計算期末有效保單數
    policies_eoy <- policies_boy * (1 - mortality_rate) * (1 - surrender_rate)
    cf_projection$policies_end_of_year[current_row] <- policies_eoy
  }
  
  return(cf_projection)
}


# ============================================================================
# 模組 2: 折現因子計算
# ============================================================================

#' 計算折現因子
#' 
#' 基於即期利率曲線計算各期的折現因子
#' 
#' @param spot_rates 各期即期利率向量
#' @return 折現因子向量
#' @export
calculate_discount_factors <- function(spot_rates) {
  proj_years <- length(spot_rates)
  years <- 1:proj_years
  discount_factors <- 1 / (1 + spot_rates)^years
  return(discount_factors)
}


# ============================================================================
# 模組 3: 風險調整計算（資本成本法）
# ============================================================================

#' 決定風險調整
#' 
#' 使用資本成本法 (Cost of Capital Approach) 計算非金融風險調整
#' 符合 IFRS 17 與 TW-ICS 99.5% 信賴水準要求
#' 
#' @param cashflow_proj 現金流量預測資料框
#' @param discount_factors 折現因子向量
#' @param assumptions 包含資本百分比與資本成本率的假設列表
#' @return 包含風險調整總額與所需資本預測的列表
#' @export
determine_risk_adjustment <- function(cashflow_proj, discount_factors, assumptions) {
  
  proj_years <- nrow(cashflow_proj) - 1
  required_capital <- numeric(proj_years)
  
  # 逐年計算未來現金流出之現值
  for (t in 1:proj_years) {
    # 取得未來現金流出
    future_outflows <- cashflow_proj$death_outflow[(t + 1):(proj_years + 1)] +
                       cashflow_proj$surrender_outflow[(t + 1):(proj_years + 1)] +
                       cashflow_proj$expense_outflow[(t + 1):(proj_years + 1)]
    
    # 計算相對於 t 時點的折現因子
    if (t <= length(discount_factors)) {
      df_t_onwards <- discount_factors[t:proj_years]
      df_relative_to_t <- df_t_onwards / discount_factors[t]
      
      # 計算未來流出現值
      pv_future_outflows_at_t <- sum(future_outflows * df_relative_to_t)
      
      # 估計所需資本 (基於 99.5% 信賴水準校準)
      required_capital[t] <- pv_future_outflows_at_t * assumptions$capital_pct
    }
  }
  
  # 計算資本成本流
  cost_of_capital_stream <- required_capital * assumptions$coc_rate
  
  # 折現回 t=0
  pv_cost_of_capital <- sum(cost_of_capital_stream * discount_factors)
  
  return(list(
    RA = pv_cost_of_capital,
    RC_Projection = required_capital
  ))
}


# ============================================================================
# 模組 4: BBA 組件計算與 CSM 推導
# ============================================================================

#' 計算 BBA 各項組件並推導 CSM
#' 
#' 整合所有模組，計算履行現金流量現值、風險調整與合約服務邊際
#' 
#' @param policy_data 保單資料列表
#' @param assumptions 精算與經濟假設列表
#' @param proj_years 預測年期數
#' @return 包含 PVFCF, RA, CSM, LRC 等關鍵指標的列表
#' @export
calculate_BBA_components <- function(policy_data, assumptions, proj_years) {
  
  # 步驟 1: 預測現金流量
  cashflow_proj <- project_cashflows(policy_data, assumptions, proj_years)
  
  # 步驟 2: 計算折現因子
  spot_rates <- assumptions$discount_curve[1:proj_years]
  discount_factors <- calculate_discount_factors(spot_rates)
  
  # 步驟 3: 計算履行現金流量現值 (PVFCF)
  net_cfs <- cashflow_proj$net_cashflow[2:(proj_years + 1)]
  pvfcf <- sum(net_cfs * discount_factors)
  
  # 步驟 4: 計算風險調整 (RA)
  ra_result <- determine_risk_adjustment(cashflow_proj, discount_factors, assumptions)
  ra <- ra_result$RA
  
  # 步驟 5: 計算 CSM（處理虧損性合約）
  fulfilment_value <- pvfcf + ra
  
  if (fulfilment_value > 0) {
    # 虧損性合約
    csm <- 0
    day_one_loss <- fulfilment_value
  } else {
    # 盈利性合約
    csm <- -fulfilment_value
    day_one_loss <- 0
  }
  
  # 彙總結果
  bba_results <- list(
    PVFCF = pvfcf,
    RA = ra,
    CSM = csm,
    Day1_Loss = day_one_loss,
    LRC = pvfcf + ra + csm,
    CashflowProjection = cashflow_proj,
    DiscountFactors = discount_factors
  )
  
  return(bba_results)
}


# ============================================================================
# 模組 5: CSM 攤銷計算
# ============================================================================

#' CSM 攤銷計算
#' 
#' 計算 CSM 在保障期間內的系統性攤銷與釋放
#' 
#' @param initial_csm 初始 CSM 金額
#' @param cashflow_proj 現金流量預測資料框
#' @param policy_data 保單資料列表
#' @param assumptions 假設列表（含鎖定利率）
#' @return CSM 攤銷時間表資料框
#' @export
amortize_csm <- function(initial_csm, cashflow_proj, policy_data, assumptions) {
  
  proj_years <- nrow(cashflow_proj) - 1
  
  # 初始化攤銷表
  amortization_schedule <- tibble(
    year = 1:proj_years,
    csm_opening = 0,
    interest_accretion = 0,
    csm_release = 0,
    csm_closing = 0
  )
  
  # 計算保障單位（Coverage Units）
  coverage_units <- cashflow_proj$policies_start_of_year[2:(proj_years + 1)] * 
                    policy_data$sum_assured_per_policy
  
  # 設定初始 CSM
  amortization_schedule$csm_opening[1] <- initial_csm
  
  # 鎖定利率（使用第一年折現率）
  locked_in_rate <- assumptions$discount_curve[1]
  
  # 逐年攤銷
  for (t in 1:proj_years) {
    # 利息增值
    interest <- amortization_schedule$csm_opening[t] * locked_in_rate
    amortization_schedule$interest_accretion[t] <- interest
    
    # 計算攤銷金額
    csm_balance_after_interest <- amortization_schedule$csm_opening[t] + interest
    
    # 計算剩餘保障單位
    remaining_coverage_units <- sum(coverage_units[t:proj_years])
    
    if (remaining_coverage_units > 0) {
      release_amount <- (coverage_units[t] / remaining_coverage_units) * 
                        csm_balance_after_interest
    } else {
      release_amount <- 0
    }
    amortization_schedule$csm_release[t] <- release_amount
    
    # 期末 CSM
    closing_balance <- csm_balance_after_interest - release_amount
    amortization_schedule$csm_closing[t] <- closing_balance
    
    # 設定下期期初
    if (t < proj_years) {
      amortization_schedule$csm_opening[t + 1] <- closing_balance
    }
  }
  
  return(amortization_schedule)
}


# ============================================================================
# 模組 6: 敏感性分析框架
# ============================================================================

#' 運行單一場景分析
#' 
#' @param scenario_name 情境名稱
#' @param policy_data 保單資料
#' @param assumptions 基準假設
#' @param assumption_modifier 修改假設的函數
#' @param proj_years 預測年期
#' @return 場景分析結果資料框
#' @export
run_scenario <- function(scenario_name, policy_data, assumptions, 
                        assumption_modifier, proj_years) {
  
  # 修改假設
  scenario_assumptions <- assumption_modifier(assumptions)
  
  # 運行 BBA 計算
  results <- calculate_BBA_components(policy_data, scenario_assumptions, proj_years)
  
  # 回傳結果
  data.frame(
    Scenario = scenario_name,
    PVFCF = results$PVFCF,
    RA = results$RA,
    CSM = results$CSM,
    LRC = results$LRC
  )
}


# ============================================================================
# 輔助函數：格式化輸出
# ============================================================================

#' 格式化財務數字（百萬元）
#' @param x 數值向量
#' @return 格式化字串
format_millions <- function(x) {
  paste0("NT$ ", formatC(x / 1e6, format = "f", digits = 1, big.mark = ","), "M")
}

#' 列印 BBA 計算摘要
#' @param bba_results BBA 計算結果列表
print_bba_summary <- function(bba_results) {
  cat("\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("IFRS 17 建築區塊法 (BBA) 計算結果摘要\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat(sprintf("履行現金流量現值 (PVFCF): %s\n", format_millions(bba_results$PVFCF)))
  cat(sprintf("風險調整 (RA):            %s\n", format_millions(bba_results$RA)))
  cat(sprintf("合約服務邊際 (CSM):       %s\n", format_millions(bba_results$CSM)))
  cat(sprintf("剩餘保障負債 (LRC):       %s\n", format_millions(bba_results$LRC)))
  if (bba_results$Day1_Loss > 0) {
    cat(sprintf("\n⚠️  第一日虧損:            %s\n", format_millions(bba_results$Day1_Loss)))
  }
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("\n")
}


# ============================================================================
# 結束模型定義
# ============================================================================

cat("IFRS 17 CSM 計算模型已載入\n")
cat("所有功能模組準備就緒\n\n")

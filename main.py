import pandas as pd
import sanitize_sba as san
import regression_sba as reg
import graphs_sba as graphs

# Import and analyze data
df = pd.read_csv('./data/SBAnational.csv');

df = san.cleanup(df);
df = san.remove_unused(df);
df = san.enhance_great_depression(df);

#graphs.corr_matrix(df);
#graphs.avg_amt_by_industry(df);
#graphs.avg_dispersement_delay(df);
#graphs.total_loan_resolutions(df);
#graphs.pif_vs_defaulted_by_disbursementFY(df);
#graphs.pif_vs_defaulted_by_real_estate(df);

reg.xgboost_regression_importance(df);

print(df);

# Save our model and use it for other SBA apps
#xgboost.save_model("./sba_default.xgb")

def compute_nbr(df):
    pos = sum(df['Sentiment'] == "Positive")
    neg = sum(df['Sentiment'] == "Negative")
    total = len(df)
    return (pos - neg) / total if total > 0 else 0


nbr_before = compute_nbr(datasets["Before_Layoff"])
nbr_after = compute_nbr(datasets["After_Layoff"])

print(f" Net Brand Reputation Before Layoff: {nbr_before:.4f}")
print(f" Net Brand Reputation After Layoff: {nbr_after:.4f}")

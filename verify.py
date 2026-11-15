from src.redaction.placeholders import repair_mojibake, strip_internal_identifiers, display_text, neutralize_placeholders

bad = "Document\u00e2\u0080\u00afID\u00e2\u0080\u00afANX\u00e2\u0080\u0091MTG\u00e2\u0080\u00910001"
r = repair_mojibake(bad)
print("1 Mojibake repair: 'â' absent:", "â" not in r)

text = "The file (Document ID: ANX-MTG-0001) is at 1847 Willow <PERSON>, TX 78745."
s = strip_internal_identifiers(text)
print("2 Strip IDs: 'ANX-MTG-0001' absent:", "ANX-MTG-0001" not in s)
print("   '<PERSON>' present:", "<PERSON>" in s)

d = display_text(text)
print("3 Display text: 'ANX-MTG-0001' absent:", "ANX-MTG-0001" not in d)
print("   '1847' absent:", "1847" not in d)
print("   '78745' absent:", "78745" not in d)
print("   '<PERSON>' absent:", "<PERSON>" not in d)
print("   'property address' present:", "property address" in d)

m = display_text("**Loan terms**\nThe rate is 7.25%.")
print("4 Markdown bold:", m.startswith("**Loan terms**"))

n = neutralize_placeholders("The rate is 7.25%.")
print("5 Neutralize no placeholders:", n == "The rate is 7.25%")

p = display_text("Borrower <PERSON> holds account <IN_ACCOUNT_NO>, loan <IN_LOAN_NO>.")
print("6 Display with placeholders:", "the borrower" in p, "the account number" in p, "the loan number" in p)
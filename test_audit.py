import copy,json,tempfile,unittest
from pathlib import Path
from audit import repairs,audit,load,valid
class AuditTests(unittest.TestCase):
 def setUp(self):
  self.y=dict(category='refund',priority='P3',amount_paise=None,txn_date=None,txn_id=None,channel=None,language='en',needs_human=False)
 def fix(self,t):return repairs(self.y,'Received: 2026-01-01 (Thu)\nSource: email\n\n'+t)[0]
 def test_money(self):
  for text,want in [('Total INR 397 ka tha.',39700),('Amount: Rs. 24566 Kindly resolve.',2456600),('It was for 1.3 lakhs.',13000000),('The transaction amount is Rs 497.99.',49799),('It was for 34.3k.',3430000)]:
   with self.subTest(text=text):self.assertEqual(self.fix(text)['amount_paise'],want)
 def test_distractors(self):
  self.assertIsNone(self.fix('FYI my daily limit is 3.4k.')['amount_paise'])
  self.assertEqual(self.fix('The transaction amount is INR 897. FYI my daily limit is Rs 37,621.50.')['amount_paise'],89700)
 def test_dates_and_quotes(self):
  self.assertEqual(self.fix('That was day before yesterday.\n> yesterday I called RBI')['txn_date'],'2025-12-30')
  self.assertEqual(self.fix('Refund missing.\n> RBI legal action')['priority'],'P3')
 def test_threshold(self):
  for n,p in [(4999,'P3'),(5000,'P2'),(49999,'P2'),(50000,'P1')]:self.assertEqual(self.fix(f'The amount was Rs {n}.')['priority'],p)
 def test_nontransaction(self):
  self.y['category']='kyc';self.y['txn_id']='NP1234567890';self.assertIsNone(self.fix('It was for 50k.')['amount_paise'])
 def test_schema(self):
  self.y['needs_human']=1;self.assertFalse(valid(self.y))
 def test_incomplete_not_rebuilt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);r={'id':'tr-x','messages':[{'role':'system','content':'s'},{'role':'user','content':'t'},{'role':'assistant','content':'{"category":'}]};(p/'train').write_text(json.dumps(r)+'\n');audit(p/'train',p/'out');self.assertEqual(load(p/'out/train_clean.jsonl'),[])
if __name__=='__main__':unittest.main()

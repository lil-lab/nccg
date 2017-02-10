package edu.uw.cs.lil.amr.lambda.convert;

import org.junit.Assert;
import org.junit.Test;

import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.amr.TestServices;

public class LogicalExpressionToAmrTest {

	public LogicalExpressionToAmrTest() {
		TestServices.init();
	}

	@Test
	public void test() {
		final LogicalExpression exp = TestServices
				.getCategoryServices()
				.readSemantics(
						"(a:<id,<<e,t>,e>> !1 (lambda $0:e (and:<t*,t>\n"
								+ "	(attack-01:<e,t> $0)\n"
								+ "	(c_REL-of:<e,<e,t>> $0 \n"
								+ "		(a:<id,<<e,t>,e>> !2 (lambda $1:e (and:<t*,t>\n"
								+ "			(c_REL:<e,<e,t>> $1 \n"
								+ "				(a:<id,<<e,t>,e>> !3 (lambda $2:e (and:<t*,t>\n"
								+ "					(c_ARG1:<e,<e,t>> $2 \n"
								+ "						(ref:<id,e> na:id))\n"
								+ "					(capital:<e,t> $2)))))\n"
								+ "			(war-01:<e,t> $1)\n"
								+ "			(c_REL:<e,<e,t>> $1 \n"
								+ "				(a:<id,<<e,t>,e>> !4 (lambda $3:e (and:<t*,t>\n"
								+ "					(country:<e,t> $3)\n"
								+ "					(c_name:<e,<e,t>> $3 \n"
								+ "						(a:<id,<<e,t>,e>> !5 (lambda $4:e (and:<t*,t>\n"
								+ "							(name:<e,t> $4)\n"
								+ "							(c_op:<e,<txt,t>> $4 Estonia:txt)))))))))))))\n"
								+ "	(c_mod:<e,<e,t>> $0 \n"
								+ "		(a:<id,<<e,t>,e>> !6 (lambda $5:e (cyber:<e,t> $5)))))))");
		final String amr = LogicalExpressionToAmr.of(exp, true);
		Assert.assertEquals("(a / attack-01\n" + "   :REL-of (w / war-01\n"
				+ "     :REL (c / capital\n" + "       :ARG1 unk0)\n"
				+ "     :REL (c2 / country\n" + "       :name (n / name\n"
				+ "         :op1 \"Estonia\")))\n" + "   :mod (c3 / cyber))",
				amr);

	}

	@Test
	public void test2() {
		final LogicalExpression exp = TestServices
				.getCategoryServices()
				.readSemantics(
						"(a:<id,<<e,t>,e>> !1 (lambda $0:e (and:<t*,t>\n"
								+ "	(SLOPPY:<e,t> $0)\n"
								+ "	(c_REL:<e,<e,t>> $0 \n"
								+ "		(a:<id,<<e,t>,e>> !2 (lambda $1:e (and:<t*,t>\n"
								+ "			(group:<e,t> $1)\n"
								+ "			(c_REL:<e,<e,t>> $1 \n"
								+ "				(a:<id,<<e,t>,e>> !3 (lambda $2:e (and:<t*,t>\n"
								+ "					(asset:<e,t> $2)\n"
								+ "					(c_REL:<e,<e,t>> $2 \n"
								+ "						(a:<id,<<e,t>,e>> !4 (lambda $3:e (freeze-02:<e,t> $3))))\n"
								+ "					(c_ARG1:<e,<e,t>> $2 \n"
								+ "						(a:<id,<<e,t>,e>> !5 (lambda $4:e (and:<t*,t>\n"
								+ "							(and:<e,t> $4)\n"
								+ "							(c_op1:<e,<e,t>> $4 \n"
								+ "								(a:<id,<<e,t>,e>> !6 (lambda $5:e (individual:<e,t> $5))))\n"
								+ "							(c_op2:<e,<e,t>> $4 \n"
								+ "								(a:<id,<<e,t>,e>> !7 (lambda $6:e (group:<e,t> $6))))))))))))\n"
								+ "			(c_REL:<e,<e,t>> $1 \n"
								+ "				(a:<id,<<e,t>,e>> !8 (lambda $7:e (and:<t*,t>\n"
								+ "					(SLOPPY:<e,t> $7)\n"
								+ "					(c_REL:<e,<e,t>> $7 \n"
								+ "						(a:<id,<<e,t>,e>> !9 (lambda $8:e (and:<t*,t>\n"
								+ "							(c_REL:<e,<e,t>> $8 \n"
								+ "								(a:<id,<<e,t>,e>> !10 (lambda $9:e (and:<t*,t>\n"
								+ "									(SLOPPY:<e,t> $9)\n"
								+ "									(c_REL:<e,<e,t>> $9 -:e)))))\n"
								+ "							(state-01:<e,t> $8)\n"
								+ "							(c_ARG1:<e,<e,t>> $8 \n"
								+ "								(ref:<id,e> !5))))))\n"
								+ "					(c_REL:<e,<e,t>> $7 \n"
								+ "						(a:<id,<<e,t>,e>> !11 (lambda $10:e (and:<t*,t>\n"
								+ "							(c_ARG0:<e,<e,t>> $10 \n"
								+ "								(ref:<id,e> !12))\n"
								+ "							(c_ARG1:<e,<e,t>> $10 \n"
								+ "								(a:<id,<<e,t>,e>> !13 (lambda $11:e (and:<t*,t>\n"
								+ "									(law:<e,t> $11)\n"
								+ "									(c_REL:<e,<e,t>> $11 \n"
								+ "										(a:<id,<<e,t>,e>> !14 (lambda $12:e (and:<t*,t>\n"
								+ "											(c_name:<e,<e,t>> $12 \n"
								+ "												(a:<id,<<e,t>,e>> !15 (lambda $13:e (and:<t*,t>\n"
								+ "													(name:<e,t> $13)\n"
								+ "													(c_op:<e,<txt,t>> $13 Europe:txt)))))\n"
								+ "											(continent:<e,t> $12)))))))))\n"
								+ "							(state-01:<e,t> $10)))))))))))))\n"
								+ "	(c_REL:<e,<e,t>> $0 \n"
								+ "		(a:<id,<<e,t>,e>> !12 (lambda $14:e (and:<t*,t>\n"
								+ "			(organization:<e,t> $14)\n"
								+ "			(c_name:<e,<e,t>> $14 \n"
								+ "				(a:<id,<<e,t>,e>> !16 (lambda $15:e (and:<t*,t>\n"
								+ "					(c_op:<e,<txt,t>> $15 United++Nations:txt)\n"
								+ "					(name:<e,t> $15))))))))))))");
		final String amr = LogicalExpressionToAmr.of(exp, true);
		Assert.assertEquals("(S / SLOPPY\n" + "   :REL (g / group\n"
				+ "     :REL (a / asset\n" + "       :REL (f / freeze-02)\n"
				+ "       :ARG1 (a2 / and\n"
				+ "         :op1 (i / individual)\n"
				+ "         :op2 (g2 / group)))\n" + "     :REL (S2 / SLOPPY\n"
				+ "       :REL (s / state-01\n"
				+ "         :REL (S3 / SLOPPY\n" + "           :REL -)\n"
				+ "         :ARG1 a2)\n" + "       :REL (s2 / state-01\n"
				+ "         :ARG0 x\n" + "         :ARG1 (l / law\n"
				+ "           :REL (c / continent\n"
				+ "             :name (n / name\n"
				+ "               :op1 \"Europe\"))))))\n"
				+ "   :REL (x / organization\n" + "     :name (n2 / name\n"
				+ "       :op1 \"United\"\n" + "       :op2 \"Nations\")))",
				amr);
	}
	
	public static void main(String[] args) throws Exception {
		
		String lgexp = "(a:<id,<<e,t>,e>> !1 (lambda $0:e (and:<t*,t> (head:<e,t> $0)" + 
					   " (c_op1:<e,<e,t>> $0 (a:<id,<<e,t>,e>> !2 (lambda $1:e (and:<t*,t> (insist-01:<e,t> $1)" + 
				       " (c_ARG0:<e,<e,t>> $1 (a:<id,<<e,t>,e>> !3 (lambda $2:e (and:<t*,t> (government-organization:<e,t> $2) " + 
					   "(c_ARG0-of:<e,<e,t>> $2 (a:<id,<<e,t>,e>> !4 (lambda $3:e (govern-01:<e,t> $3)))))))) " + 
				       "(c_ARG1:<e,<e,t>> $1 (a:<id,<<e,t>,e>> !5 (lambda $4:e (and:<t*,t> (support-01:<e,t> $4) " +
					   "(c_ARG0:<e,<e,t>> $4 (a:<id,<<e,t>,e>> !6 (lambda $5:e (reserve:<e,t> $5)))) (c_ARG1:<e,<e,t>> $4 " + 
				       "(a:<id,<<e,t>,e>> !7 (lambda $6:e (and:<t*,t> (force:<e,t> $6) (c_ARG1-of:<e,<e,t>> $6 " + 
					   "(a:<id,<<e,t>,e>> !8 (lambda $7:e (arm-01:<e,t> $7)))))))))))))))) (c_op2:<e,<e,t>> $0 " + 
				       "(a:<id,<<e,t>,e>> !9 (lambda $8:e (and:<t*,t> (UNK:<e,t> $8) (c_ARG1:<e,<e,t>> $8 (a:<id,<<e,t>,e>> !10 "+
					   "(lambda $9:e (and:<t*,t> (oppose-01:<e,t> $9) (c_ARG1:<e,<e,t>> $9 (a:<id,<<e,t>,e>> !11 (lambda $10:e " +
				       "(and:<t*,t> (person:<e,t> $10) (c_name:<e,<e,t>> $10 (a:<id,<<e,t>,e>> !12 (lambda $11:e (and:<t*,t> " + 
					   "(name:<e,t> $11) (c_op:<e,<txt,t>> $11 Chavez:txt))))))))) (c_mod:<e,<e,t>> $9 (a:<id,<<e,t>,e>> !13 " + 
				       "(lambda $12:e (possible:<e,t> $12)))))))) (c_ARG0:<e,<e,t>> $8 (a:<id,<<e,t>,e>> !14 (lambda $13:e (and:<t*,t> (seat:<e,t> $13) (c_prep-per:<e,<e,t>> $13 (a:<id,<<e,t>,e>> !15 (lambda $14:e (and:<t*,t> (include-91:<e,t> $14) " + 
					   "(c_op:<e,<txt,t>> $14 \"1/5\":txt))))))))))))) " + 
				       "(c_op3:<e,<e,t>> $0 (a:<id,<<e,t>,e>> !16 (lambda $15:e (coup:<e,t> $15)))))))";
		
		final LogicalExpression exp = TestServices
				.getCategoryServices()
				.readSemantics(lgexp);
		final String amr = LogicalExpressionToAmr.of(exp, true);
		System.out.println("AMR is " +amr);
	}

}

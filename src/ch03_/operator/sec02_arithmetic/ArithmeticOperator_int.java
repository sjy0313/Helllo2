
package ch03_.operator.sec02_arithmetic;

public class ArithmeticOperator_int {
	public static void main(String[] args) {
		
		int x = 10;
		int y = 3;
		int z = x % y;
		
		System.out.printf("나머지 : (%d) = (%d) %% (%d)\n", z, x, y);
		
		double s = Math.IEEEremainder(x, y);
		
		System.out.printf("IEEEremainder : (%f) = (%d) %% (%d)\n", s, x, y);
		
		int t = x / y;
		int k = x - (t * y);
		System.out.printf("몫(%d), 나머지(%d)\n", t, k);
	}
}

/* 나머지 : (1) = (10) % (3)
IEEEremainder : (1.000000) = (10) % (3)
몫(3), 나머지(1)*/

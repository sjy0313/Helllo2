
package ch03_.operator.sec01_increase_decrease;

public class SignOperatorExample {
	public static void main(String[] args) {
		// byte b = 100; byte result = -b; -> compile error 발생
		int x = -100;
		x = -x;
		System.out.println("x: " + x); // 100

		byte b = 100;
		int y = -b;
		System.out.println("y: " + y); // -100
	}
}
package ch06_.classes.sec10.static_member.exam01.static_field_method;

public class CalculatorExample {
	public static void main(String[] args) {
		double result1 = 10 * 10 * Calculator.pi; 
		int result2 = Calculator.plus(10, 5);
		int result3 = Calculator.minus(10, 5);

		System.out.println("Area of Circle : " + result1); // result1 : 314.159
		System.out.println("Addition : " + result2); // result2 : 15
		System.out.println("Subtraction : " + result3); // result3 : 5
	}
}

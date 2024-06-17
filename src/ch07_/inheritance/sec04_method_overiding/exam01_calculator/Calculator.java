package ch07_.inheritance.sec04_method_overiding.exam01_calculator;

public class Calculator {
	//메소드 선언
	// 원 넓이를 구하는 메소드를 가지고 있지만 원주율 파이가 정확하지 
	// 않기 떄문에 자식 클래스인 Compute r에서 overriding 을 해서 좀 더
	//정확한 파이를 Math.PI 통해 구함. 
	public double areaCircle(double r) {
		System.out.println("Calculator 객체의 areaCircle() 실행");
		return 3.14159 * r * r;
	}
}
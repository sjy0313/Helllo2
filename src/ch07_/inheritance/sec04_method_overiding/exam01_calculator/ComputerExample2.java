package ch07_.inheritance.sec04_method_overiding.exam01_calculator;

public class ComputerExample2 {
	public static void main(String[] args) {
		int r = 10;
		// 자식 클래스로 생성하여 부모 타입으로 받음
		Calculator calc = new Computer();
		System.out.println("원 면적: " + calc.areaCircle(r));
		
		// 부모 클래스로 생성하여 자식 타입으로 받음? - 못받음

		//Type mismatch: cannot convert from Calculator to Computer
		
		Computer comp = new Calculator();
		System.out.println("원 면적: " + calc.areaCircle(r));
	}
}

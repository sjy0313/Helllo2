package ch07_.inheritance.sec04_method_overiding.exam01_calculator;

public class ComputerExample {
	public static void main(String[] args) {
		int r = 10;

		Calculator calculator = new Calculator();
		System.out.println("원 면적: " + calculator.areaCircle(r));
		System.out.println();
		

		Computer computer = new Computer();
		System.out.println("원 면적: " + computer.areaCircle(r));
	}
}
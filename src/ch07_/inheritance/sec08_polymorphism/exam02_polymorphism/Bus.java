package ch07_.inheritance.sec08_polymorphism.exam02_polymorphism;

public class Bus extends Vehicle {
	//메소드 재정의(오버라이딩)
	@Override
	public void run() {
		System.out.println("버스가 달립니다.");
	}
}
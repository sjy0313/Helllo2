package ch07_.inheritance.sec08_polymorphism.exam01_field;

public class KumhoTire extends Tire {
	//메소드 재정의(오버라이딩)
	@Override
	public void roll() {
		System.out.println("금호 타이어가 회전합니다.");
	}
}
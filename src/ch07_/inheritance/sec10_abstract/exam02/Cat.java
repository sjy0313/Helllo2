package ch07_.inheritance.sec10_abstract.exam02;

public class Cat extends Animal {
	//추상 메소드 재정의
	@Override
	public void sound() {
		System.out.println("야옹");
	}
}
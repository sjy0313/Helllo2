package ch07_.inheritance.sec07_typecasting.exam02_promotion;

public class Child extends Parent {
	//메소드 오버라이딩
	@Override
	public void method2() {
		System.out.println("Child-method2()");
	}

	//메소드 선언 : 새로운 메소드
	public void method3() {
		System.out.println("Child-method3()");
	}
}
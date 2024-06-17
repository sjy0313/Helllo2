package ch07_.inheritance.sec04_method_overiding.exam01_calculator;
/*
 * 메소드 오버라이딩 규칙 :
 * - 부모 메소드의 선언부와 동일 : 리턴타입, 메소드이름, 파라미터 
 * - 접근제한 더 강하게 오버라이딩 할 수 없다(public -> private 불가)
 * - 새로운 예외를 throw할 수 없다
 */
public class Computer extends Calculator {
	//메소드 오버라이딩 : 어떤 메소드는 자식 클래스가 사용하기에 적합하지 
	// 않을 수 있음 이러한 메소드는 클래스에서 재정의 필요
	
	// 자식이 받은 상속받은 클래스에서 교체
	@Override // 설명자
	public double areaCircle(double r) {
		System.out.println("Computer 객체의 areaCircle() 실행");
		return Math.PI * r * r;
	}
}
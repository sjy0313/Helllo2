package ch07_.inheritance.sec10_abstract.exam01;

/*
 * 추상 클래스 : abstract class 
 * - 인스턴스화 할 수 없다(new할 수 없다)
 * - 구현 클래스를 통해서만 인스턴스화 할 수 있다
 * 		(상속 클래스를 만들어야 함)
 * 즉 실체가 있어야함.[실체간 공통되는 특성을 추출하는 것]
 * 추상 메소드는 가질 수 있다 / 추상 메소드는 구현이 없는 메소드
 */
public class SmartPhone extends Phone {
	//생성자 선언
	SmartPhone(String owner) {
		//Phone 생성자 호출
		super(owner);
	}
	

	//메소드 선언
	
	void internetSearch() {
		System.out.println("인터넷 검색을 합니다.");
	}
}
package ch07_.inheritance.sec03_parent_constructor.exam03;

public class SmartPhoneExample {
	
	public static void main(String[] args) {
		//SmartPhone 객체 생성
		//SmartPhone myPhone = new SmartPhone("갤럭시", "은색");
		SmartPhone myPhone = new SmartPhone();
		
		//Phone으로부터 상속 받은 필드 읽기
		//속성이 private으로 지정되었기 때문에 외부에서 직접 접근 불가
		// 해결방법 :
		// 부모클래스인 phone 에서 getter 와 setter 지정 
		// or, 부모클래스에서 'protected' 필드로 변경 protected String model;
		// 하위 클래스에 접근하도록 해야함.
		System.out.println("모델: " + myPhone.getModel());
		System.out.println("색상: " + myPhone.getColor()); 
	}
}
/* SmartPhone 클래스에서 Phone 클래스의 필드에 접근하려면
 *  setModel(), setColor(), getModel(), getColor() 메서드를 사용해야 합니다.
 *  이렇게 함으로써 The field Phone.model is not visible 오류를 피할 수 있고,
 *   객체 지향 프로그래밍의 원칙인 캡슐화를 잘 준수할 수 있습니다.
 */

